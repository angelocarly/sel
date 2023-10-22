use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassType, CommandBufferUsage, RenderPassBeginInfo, SecondaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Queue;
use vulkano::image::StorageImage;
use vulkano::image::view::ImageView;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{RenderPass, Subpass};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
            #version 460

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

            void main() {
                vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
                vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

                vec2 z = vec2(0.0, 0.0);
                float i;
                for (i = 0.0; i < 1.0; i += 0.005) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + c.x,
                        z.y * z.x + z.x * z.y + c.y
                    );

                    if (length(z) > 4.0) {
                        break;
                    }
                }

                vec4 to_write = vec4(vec3(i), 1.0);
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
            }
        "
    }
}

pub struct ComputeRaysPipeline {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pipeline: Arc<ComputePipeline>
}

impl ComputeRaysPipeline {
    pub fn new(
        gfx_queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> ComputeRaysPipeline {
        let device = gfx_queue.device();

        let cs = cs::load(device.clone())
            .expect("Failed to create shader module.");

        let pipeline = ComputePipeline::new(
            device.clone(),
            cs.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        ).expect("Failed to create compute pipeline.");

        ComputeRaysPipeline {
            gfx_queue,
            render_pass,
            command_buffer_allocator,
            pipeline
        }
    }

    fn create_descriptor_set(&self, image_view: Arc<ImageView<StorageImage>>) -> Arc<PersistentDescriptorSet>
    {
        let device = self.gfx_queue.device();
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let pipeline_layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let descriptor_set_layout_index = 0;
        PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            pipeline_layout.clone(),
            [WriteDescriptorSet::image_view(0, image_view.clone())],
        ).unwrap()

    }

    pub fn draw(
        &self,
        image_view: Arc<ImageView<StorageImage>>
    ) -> SecondaryAutoCommandBuffer {
        let mut builder = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator.as_ref(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                ..Default::default()
            },
        ).unwrap();

        let descriptor_set = self.create_descriptor_set(image_view);

        builder.bind_pipeline_compute(self.pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            self.pipeline.layout().clone(),
            0,
            descriptor_set
        )
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap();

        builder.build().unwrap()
    }
}

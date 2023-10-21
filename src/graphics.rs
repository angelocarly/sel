use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{CommandBufferBuilderAlloc, StandardCommandBufferAllocator};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet};

use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::shader::ShaderModule;

/**
 * For now: Contains logic for rendering an image with a compute shader
 */

// ================================== Image ========================================================
pub fn get_image(memory_allocator: &StandardMemoryAllocator, queue: Arc<Queue>) -> (Arc<StorageImage>, Arc<ImageView<StorageImage>>) {
    let image = StorageImage::new(
        memory_allocator,
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    ).unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();

    return (image, view);
}

// ================================== Compute ======================================================

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

pub fn get_pipeline(device: Arc<Device>, view: Arc<ImageView<StorageImage>>) -> (Arc<ComputePipeline>, Arc<PersistentDescriptorSet>) {

    let compute_pipeline = {

        let cs = cs::load(device.clone())
            .expect("Failed to create shader module.");

        ComputePipeline::new(
            device.clone(),
            cs.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        ).expect("Failed to create compute pipeline.")
    };

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let pipeline_layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_set_layout_index = 0;
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        pipeline_layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())],
    ).unwrap();

    return (compute_pipeline, descriptor_set);
}

// ================================== Graphics =====================================================

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            layout( location = 0 ) out vec2 outUV;

            void main()
            {
                outUV = vec2( ( gl_VertexIndex << 1 ) & 2, gl_VertexIndex & 2 );
                gl_Position = vec4( outUV * 2.0f + -1.0f, 0.0f, 1.0f );
            }
        "
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 460

            layout( location = 0 ) in vec2 inUV;

            layout( location = 0 ) out vec4 f_color;

            layout(set = 0, binding = 0, rgba8) uniform readonly image2D inImg;

            void main() {
                f_color = imageLoad( inImg, ivec2(inUV) ).rgba;
            }
        ",
    }
}

pub fn get_graphics_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    view: Arc<ImageView<StorageImage>>,
) -> (Arc<GraphicsPipeline>, Arc<PersistentDescriptorSet>) {
    let graphics_pipeline = GraphicsPipeline::start()
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device.clone())
        .unwrap();

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let pipeline_layout = graphics_pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_set_layout_index = 0;
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        pipeline_layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())],
    ).unwrap();

    return (graphics_pipeline, descriptor_set);
}

// ====================================== Command Buffers ==========================================

pub fn build_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    compute_pipeline: &Arc<ComputePipeline>,
    compute_descriptor_set: Arc<PersistentDescriptorSet>,
    graphics_pipeline: &Arc<GraphicsPipeline>,
    graphics_descriptor_set: Arc<PersistentDescriptorSet>,
    framebuffers: &Vec<Arc<Framebuffer>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit, // don't forget to write the correct buffer usage
            )
                .unwrap();

            builder
                // Dispatch the compute shader
                .bind_pipeline_compute(compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0,
                    compute_descriptor_set.clone()
                )
                .dispatch([1024 / 8, 1024 / 8, 1])
                .unwrap()

                // Render the compute result
                .bind_pipeline_graphics(graphics_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    graphics_pipeline.layout().clone(),
                    0,
                    graphics_descriptor_set.clone()
                )
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .draw(3, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

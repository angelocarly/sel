use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferInheritanceRenderPassType, CommandBufferUsage, RenderPassBeginInfo, SecondaryAutoCommandBuffer};
use vulkano::device::Queue;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{RenderPass, Subpass};

mod vs {
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

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 460

            layout( location = 0 ) in vec2 inUV;

            layout( location = 0 ) out vec4 f_color;

            // layout(set = 0, binding = 0, rgba8) uniform readonly image2D inImg;

            void main() {
                f_color = vec4( 1.0f, 0.0f, 0.0f, 1.0f );
                // f_color = imageLoad( inImg, ivec2(inUV) ).rgba;
            }
        ",
    }
}

pub struct DrawPipeline {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pipeline: Arc<GraphicsPipeline>
}

impl DrawPipeline {
    pub fn new(
        gfx_queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> DrawPipeline {
        let device = gfx_queue.device();

        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();

        let pipeline = GraphicsPipeline::start()
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();

        DrawPipeline {
            gfx_queue,
            render_pass,
            command_buffer_allocator,
            pipeline
        }
    }

    fn create_descriptor_set() {

    }

    pub fn draw(
        &self,
        viewport: &Viewport
    ) -> SecondaryAutoCommandBuffer {
        let mut builder = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator.as_ref(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(Subpass::from(self.render_pass.clone(), 0).unwrap().into()),
                ..Default::default()
            },
        ).unwrap();

        builder.set_viewport(0,[viewport.clone()]);
        builder.bind_pipeline_graphics(self.pipeline.clone());
        builder.draw(3, 1, 0, 0).unwrap();

        builder.build().unwrap()
    }
}
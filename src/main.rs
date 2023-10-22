pub mod vulkan;
mod draw_pipeline;
mod compute_rays_pipeline;

use std::sync::Arc;
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::swapchain;

use tracing_subscriber;
use tracing_subscriber::filter::FilterExt;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};
use vulkano::image::sys::Image;
use vulkano::image::view::ImageView;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit::window::Window;
use winit::event_loop::ControlFlow;

use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, Subpass};
use vulkano::swapchain::{AcquireError, SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;

use vulkano_win::VkSurfaceBuild;
use crate::compute_rays_pipeline::ComputeRaysPipeline;
use crate::draw_pipeline::DrawPipeline;
use crate::vulkan::get_framebuffers;

pub fn get_image(memory_allocator: &StandardMemoryAllocator, queue: Arc<Queue>) -> (Arc<StorageImage>, Arc<ImageView<StorageImage>>) {
    let mut image = StorageImage::with_usage(
        memory_allocator,
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        ImageUsage::TRANSFER_SRC
            | ImageUsage::TRANSFER_DST
            | ImageUsage::SAMPLED
            | ImageUsage::STORAGE,
        ImageCreateFlags::empty(),
        Some(queue.queue_family_index()),
    ).unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();

    return (image, view);
}

fn build_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    draw_pipeline: &Arc<DrawPipeline>,
    compute_pipeline: &Arc<ComputeRaysPipeline>,
    viewport: &Viewport,
    image_view: &Arc<ImageView<StorageImage>>
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit, // don't forget to write the correct buffer usage
            ).unwrap();

            // Execute the compute pipeline
            // TODO: Perform compute shader without access error
            // builder.execute_commands(
            //     compute_pipeline.draw(image_view.clone())
            // ).unwrap();

            // Start a renderpass for the framebuffer
            builder.begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::SecondaryCommandBuffers,
            ).unwrap();

            // Bind the pipeline
            builder.execute_commands(
                draw_pipeline.draw(viewport)
            ).unwrap();

            // End renderpass
            builder.end_render_pass().unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn main() {

    // Logging setup
    tracing_subscriber::fmt::init();

    // Vulkan setup
    let instance = vulkan::create_instance();

    // Window - creates vk surface
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let window = surface
        .object()
        .unwrap()
        .clone()
        .downcast::<Window>()
        .unwrap();

    window.set_title("Sel");

    // Vulkan devices - takes window surface
    let (physical_device, queue_family_index) = vulkan::create_physical_device(instance.clone(), surface.clone());
    let (device, mut queues) = vulkan::create_device(physical_device.clone(), queue_family_index);
    let queue = queues.next().unwrap();
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // Swapchain
    let (mut swapchain, images) = vulkan::create_swapchain(
        physical_device.clone(),
        device.clone(),
        window.clone(),
        surface.clone(),
    );

    // Render pass
    let render_pass = vulkan::get_render_pass(device.clone(), &swapchain);
    let framebuffers = get_framebuffers(&images, &render_pass);

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    // Image
    let (image, image_view) = get_image(
        &memory_allocator,
        queue.clone(),
    );

    // Command buffers
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    // Compute pipeline
    let compute_pipeline = Arc::new(
        ComputeRaysPipeline::new(
            queue.clone(),
            render_pass.clone(),
            command_buffer_allocator.clone()
        )
    );

    // Draw pipeline
    let draw_pipeline = Arc::new(
        draw_pipeline::DrawPipeline::new(
            queue.clone(),
            render_pass.clone(),
            command_buffer_allocator.clone(),
        )
    );

    let mut command_buffers = build_command_buffers(
        &command_buffer_allocator,
        &queue,
        &framebuffers,
        &draw_pipeline,
        &compute_pipeline,
        &viewport,
        &image_view
    );

    // Event loop
    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                window_resized = true;
            }
            Event::MainEventsCleared => {
                if window_resized || recreate_swapchain {
                    recreate_swapchain = false;

                    let new_dimensions = window.inner_size();

                    let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                        image_extent: new_dimensions.into(), // here, "image_extend" will correspond to the window dimensions
                        ..swapchain.create_info()
                    }) {
                        Ok(r) => r,
                        // This error tends to happen when the user is manually resizing the window.
                        // Simply restarting the loop is the easiest way to fix this issue.
                        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                        Err(e) => panic!("failed to recreate swapchain: {}", e),
                    };
                    swapchain = new_swapchain;
                    let new_framebuffers = get_framebuffers(&new_images, &render_pass);

                    if window_resized {
                        window_resized = false;

                        viewport.dimensions = new_dimensions.into();
                        command_buffers = build_command_buffers(
                            &command_buffer_allocator,
                            &queue,
                            &new_framebuffers,
                            &draw_pipeline,
                            &compute_pipeline,
                            &viewport,
                            &image_view
                        );
                    }
                }

                let (image_i, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                // Wait for the fence related to this image to finish (normally this would be the oldest fence)
                if let Some(image_fence) = &fences[image_i as usize] {
                    image_fence.wait(None).unwrap();
                }

                let previous_future = match fences[previous_fence_i as usize].clone() {
                    None => {
                        let mut now = sync::now(device.clone());
                        now.cleanup_finished();

                        now.boxed()
                    }
                    Some(fence) => fence.boxed()
                };

                let future = previous_future
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                    )
                    .then_signal_fence_and_flush();

                fences[image_i as usize] = match future {
                    Ok(value) => Some(Arc::new(value)),
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        None
                    }
                    Err(e) => {
                        panic!("Failed to flush future: {}", e);
                    }
                };

                previous_fence_i = image_i;
            }
            _ => {}
        }
    });
}


pub mod vulkan;
mod graphics;

use std::sync::Arc;
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::swapchain;

use tracing_subscriber;
use tracing_subscriber::filter::FilterExt;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit::window::Window;
use winit::event_loop::ControlFlow;

use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::swapchain::{AcquireError, SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;

use vulkano_win::VkSurfaceBuild;
use crate::graphics::{fs, vs};
use crate::vulkan::get_framebuffers;

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
    let (image, image_view) = graphics::get_image(
        &memory_allocator,
        queue.clone(),
    );

    // Compute pipeline
    let (compute_pipeline, compute_descriptor_set) = graphics::get_pipeline(
        device.clone(),
        image_view.clone()
    );

    // Graphics pipeline
    let vs = vs::load(device.clone()).expect("failed to create shader module.");
    let fs = fs::load(device.clone()).expect("failed to create shader module.");

    let (graphics_pipeline, graphics_descriptor_set) = graphics::get_graphics_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
        image_view.clone()
    );

    // Command buffers
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut command_buffers = graphics::build_command_buffers(
        &command_buffer_allocator,
        &queue,
        &compute_pipeline,
        compute_descriptor_set.clone(),
        &graphics_pipeline,
        graphics_descriptor_set.clone(),
        &framebuffers,
        &image
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
                        let (new_compute_pipeline, new_compute_descriptor_set) = graphics::get_pipeline(
                            device.clone(),
                            image_view.clone(),
                        );
                        let (new_graphics_pipeline, new_graphics_descriptor_set) = graphics::get_graphics_pipeline(
                            device.clone(),
                            vs.clone(),
                            fs.clone(),
                            render_pass.clone(),
                            viewport.clone(),
                            image_view.clone(),
                        );
                        command_buffers = graphics::build_command_buffers(
                            &command_buffer_allocator,
                            &queue,
                            &new_compute_pipeline,
                            new_compute_descriptor_set.clone(),
                            &new_graphics_pipeline,
                            new_graphics_descriptor_set.clone(),
                            &new_framebuffers,
                            &image
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


pub mod graphics;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::swapchain;

use tracing_subscriber;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit::window::Window;
use winit::event_loop::ControlFlow;

use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::swapchain::{AcquireError, SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo};

use vulkano_win::VkSurfaceBuild;
use crate::graphics::{fs, get_framebuffers, vs};

fn main() {

    // Logging setup
    tracing_subscriber::fmt::init();

    // Vulkan setup
    let instance = graphics::create_instance();

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
    let (physical_device, queue_family_index) = graphics::create_physical_device(instance.clone(), surface.clone());
    let (device, mut queues) = graphics::create_device(physical_device.clone(), queue_family_index);
    let queue = queues.next().unwrap();
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // Swapchain
    let (mut swapchain, images) = graphics::create_swapchain(
        physical_device.clone(),
        device.clone(),
        window.clone(),
        surface.clone(),
    );

    // Render pass
    let render_pass = graphics::get_render_pass(device.clone(), &swapchain);
    let framebuffers = get_framebuffers(&images, &render_pass);

    // Model
    let vertex1 = graphics::MyVertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = graphics::MyVertex {
        position: [0.0, 0.5],
    };
    let vertex3 = graphics::MyVertex {
        position: [0.5, -0.25],
    };
    let vertex_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3]
    ).unwrap();

    // Pipeline
    let vs = vs::load(device.clone()).expect("failed to create shader module.");
    let fs = fs::load(device.clone()).expect("failed to create shader module.");

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let pipeline = graphics::get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone()
    );

    // Command buffers
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut command_buffers = graphics::build_command_buffers(
        &command_buffer_allocator,
        &queue,
        &pipeline,
        &framebuffers,
        &vertex_buffer,
    );

    // Event loop
    let mut window_resized = false;
    let mut recreate_swapchain = false;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                window_resized = true;
            },
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
                        let new_pipeline = graphics::get_pipeline(
                            device.clone(),
                            vs.clone(),
                            fs.clone(),
                            render_pass.clone(),
                            viewport.clone(),
                        );
                        command_buffers = graphics::build_command_buffers(
                            &command_buffer_allocator,
                            &queue,
                            &new_pipeline,
                            &new_framebuffers,
                            &vertex_buffer,
                        );
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

                    let execution = sync::now(device.clone())
                        .join(acquire_future)
                        .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                        .unwrap()
                        .then_swapchain_present(
                            queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                        )
                        .then_signal_fence_and_flush();

                    match execution {
                        Ok(future) => {
                            future.wait(None).unwrap();  // wait for the GPU to finish
                        }
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                        }
                        Err(e) => {
                            println!("Failed to flush future: {e}");
                        }
                    }
                }
            },
            _ => {}
        }
    });

}


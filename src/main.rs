pub mod graphics;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano::pipeline::Pipeline;
use tracing_subscriber;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit::window::Window;
use winit::event_loop::ControlFlow;

use image::{ImageBuffer, Rgba};
use vulkano::image::ImageUsage;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo, SwapchainCreationError};

use vulkano_win::VkSurfaceBuild;
use winit::window::CursorIcon::Default;
use crate::graphics::{fs, get_framebuffers, vs};


fn test() {
    let mut v: Vec<i32> = vec![1, 2, 3];
    let num: &i32 = &v[1];
    v.push(4);
}



fn main() {

    test();

    // Logging setup
    tracing_subscriber::fmt::init();

    let instance = graphics::create_instance();

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

    // Vulkan setup
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

    let render_pass = graphics::get_render_pass(device.clone(), &swapchain);
    let framebuffers = get_framebuffers(&images, &render_pass);

    // Command buffer
    use std::default::Default;
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    // vertex stuffs
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

    let mut command_buffers = graphics::get_command_buffers(
        &command_buffer_allocator,
        &queue,
        &pipeline,
        &framebuffers,
        &vertex_buffer,
    );

    // Record the commands
    let buf = graphics::draw(
        device.clone(),
        memory_allocator,
        queue.clone(),
        &mut builder
    );

    // Execute
    let command_buffer = builder.build().unwrap();

    // Event loop
    let mut window_resized = false;
    let mut recreate_swapchain = false;

    event_loop.run(|event, _, control_flow| {
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
                if recreate_swapchain {
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
                        Err(e) => panic!("failed to recreate swapchain: {e}"),
                    };
                    swapchain = new_swapchain;
                    let new_framebuffers = get_framebuffers(&new_images, &render_pass);
                }
            },
            _ => {}
        }
    });

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    // Write output image
    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();

}


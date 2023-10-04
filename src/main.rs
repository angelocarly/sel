pub mod graphics;

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
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};

use vulkano_win::VkSurfaceBuild;
use winit::window::CursorIcon::Default;


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
    let caps = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("Failed to get surface capabilities.");
    let dimensions = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    let image_format = Some(
        physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..Default::default()
        },
    ).unwrap();

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

    event_loop.run(|event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
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


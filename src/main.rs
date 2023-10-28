pub mod vulkan;

use std::sync::Arc;
use image::{ImageBuffer, Rgba};
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::{descriptor_set, swapchain};

use tracing_subscriber;
use tracing_subscriber::filter::FilterExt;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyImageToBufferInfo, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassContents};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Queue;
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::{ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};
use vulkano::image::sys::Image;
use vulkano::image::view::ImageView;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};

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
use crate::vulkan::get_framebuffers;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
            #version 460
            #extension GL_EXT_shader_atomic_float : require

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform image2D accum_image;

            void main() {
                // vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));

                imageAtomicAdd(accum_image, ivec2(0), 1);
            }
        "
    }
}

pub fn get_image(memory_allocator: &StandardMemoryAllocator, queue: Arc<Queue>) -> (Arc<StorageImage>, Arc<ImageView<StorageImage>>) {
    let mut image = StorageImage::new(
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

    // Image
    let (image, image_view) = get_image(
        &memory_allocator,
        queue.clone(),
    );

    // Compute pipeline
    let cs = cs::load(device.clone())
        .expect("Failed to create shader module.");

    let pipeline = ComputePipeline::new(
        device.clone(),
        cs.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    ).expect("Failed to create compute pipeline.");

    // Descriptor set
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let pipeline_layout = pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        pipeline_layout.clone(),
        [WriteDescriptorSet::image_view(0, image_view.clone())],
    ).unwrap();

    // Buffer
    let buf = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
        .expect("Failed to create buffer.");

    // Command buffers
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    builder
        .clear_color_image(ClearColorImageInfo::image(image.clone()))
        .unwrap()
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            descriptor_set,
        )
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))
        .unwrap();


    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
}


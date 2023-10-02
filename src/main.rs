use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano_shaders::shader;
use vulkano::pipeline::{ComputePipeline, PipelineBindPoint};
use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::format::Format;
use vulkano::command_buffer::ClearColorImageInfo;
use vulkano::format::ClearColorValue;
use vulkano::image::view::ImageView;

use tracing::info;
use tracing_subscriber;

use std::env;
use std::sync::Arc;
use vulkano::shader::spirv::Instruction::All;

use image::{ImageBuffer, Rgba};

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
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
        ",
    }
}

fn create_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("No local Vulkan library found.");

    return if env::consts::OS == "macos" {
        // Enable the portability extension on macOS in order to support MoltenVK.
        let extensions = InstanceExtensions {
            khr_portability_enumeration: true,
            ..InstanceExtensions::empty()
        };
        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: extensions,
                enumerate_portability: true,
                ..Default::default()
            }
        ).expect("Failed to create a macos Vulkan instance.")
    } else {
        let extensions = InstanceExtensions {
            khr_portability_enumeration: false,
            ..InstanceExtensions::empty()
        };
        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: extensions,
                enumerate_portability: false,
                ..Default::default()
            }
        ).expect("Failed to create Vulkan instance.")
    }
}

fn create_device(instance: Arc<Instance>) -> (u32, Arc<Device>, impl ExactSizeIterator<Item=Arc<Queue>> + Sized) {
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices.")
        .next()
        .expect("No device available.");

    for family in physical_device.queue_family_properties() {
        info!("Queue Family: {:?}", family);
    }

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(index, properties)| { properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("Could not find a graphical queue family.") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
        .expect("Failed to create device.");

    return (queue_family_index, device, queues);
}

fn main() {

    // Logging setup
    tracing_subscriber::fmt::init();

    // Vulkan setup
    let instance = create_instance();

    let (queue_family_index, device, mut queues) = create_device(instance.clone());

    let queue = queues.next().unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    // Compute
    let shader = cs::load(device.clone())
        .expect("Failed to create shader module.");

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    ).expect("Failed to create compute pipeline.");

    // Execute

    let image = StorageImage::new(
        &memory_allocator,
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    ).unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let pipeline_layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_set_layout_index = 0;
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        pipeline_layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())],
    ).unwrap();

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

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
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


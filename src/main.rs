use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::sync::{self, GpuFuture};

use tracing::info;
use tracing_subscriber;

fn main() {

    // Logging setup
    tracing_subscriber::fmt::init();

    // Vulkan setup
    let library = VulkanLibrary::new().expect("No local Vulkan library found.");
    let instance = Instance::new(library, InstanceCreateInfo::default())
        .expect("Failed to create Vulkan instance.");

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

    let queue = queues.next().unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let data: i32 = 12;
    let buffer = Buffer::from_data(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        data
    )
    .expect("Failed to create buffer.");

    let a = *buffer.read().unwrap();
    info!("Buffer data: {:?}", a);

    // Buffer test
    let source_content: Vec<i32> = (0..64).collect();
    let source = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        source_content
    ).expect("Failed to create source buffer.");

    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        destination_content
    ).expect("Failed to create destination buffer.");

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    builder.copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let src_content = source.read().unwrap();
    let dst_content = destination.read().unwrap();
    assert_eq!(&*src_content, &*dst_content);

    info!("Src: {:?}", &src_content[..]);
    info!("Dst: {:?}", &dst_content[..]);
}


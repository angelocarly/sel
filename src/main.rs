use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{StandardMemoryAllocator, MemoryUsage, AllocationCreateInfo};

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

}

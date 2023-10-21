/*
 * Graphics setup and rendering code
 */

use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags, DeviceExtensions};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::{GraphicsPipeline};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::VulkanLibrary;

use winit::window::Window;

use std::env;
use std::sync::Arc;
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger, DebugUtilsMessengerCreateInfo};

use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::shader::ShaderModule;

pub(crate) fn create_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("No local Vulkan library found.");

    println!("List of Vulkan debugging layers available to use:");
    let layers = library.layer_properties().unwrap();
    for l in layers {
        println!("\t{}", l.name());
    }

    let layers = vec!["VK_LAYER_KHRONOS_validation".to_owned()];

    let mut required_extensions = vulkano_win::required_extensions(&library);

    let instance =
        if env::consts::OS == "macos" {
            // Enable the portability extension on macOS in order to support MoltenVK.
            required_extensions.khr_portability_enumeration = true;
            Instance::new(
                library,
                InstanceCreateInfo {
                    enabled_extensions: required_extensions,
                    enabled_layers: layers,
                    enumerate_portability: true,
                    ..Default::default()
                },
            ).expect("Failed to create a macos Vulkan instance.")
        } else {
            Instance::new(
                library,
                InstanceCreateInfo {
                    enabled_layers: layers,
                    enabled_extensions: required_extensions,
                    enumerate_portability: false,
                    ..Default::default()
                },
            ).expect("Failed to create Vulkan instance.")
        };

    return instance;
}

pub fn device_extensions() -> DeviceExtensions {
    return DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
}

pub fn create_physical_device(instance: Arc<Instance>, surface: Arc<Surface>) -> (Arc<PhysicalDevice>, u32) {
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices.")
        .filter(|p| p.supported_extensions().contains(&device_extensions()))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("No device available.");

    return (physical_device, queue_family_index);
}

pub fn create_device(physical_device: Arc<PhysicalDevice>, queue_family_index: u32) -> (Arc<Device>, impl ExactSizeIterator<Item=Arc<Queue>> + Sized) {
    let (device, queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions(),
            ..Default::default()
        },
    )
        .expect("Failed to create device.");

    return (device, queues);
}

pub fn create_swapchain(physical_device: Arc<PhysicalDevice>, device: Arc<Device>, window: Arc<Window>, surface: Arc<Surface>) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
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

    let (swapchain, images) = Swapchain::new(
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

    return (swapchain, images);
}

pub fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    ).unwrap()
}

pub fn get_framebuffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
                .unwrap()
        })
        .collect::<Vec<_>>()
}
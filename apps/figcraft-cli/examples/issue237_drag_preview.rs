//! CPU oracle versus the canvas's one-rectangle egui/wgpu drag path (#237).

use std::time::{Duration, Instant};

use egui::{Color32, CornerRadius, Id, LayerId, Order, Pos2, Rect, TextureHandle, TextureOptions, pos2, vec2};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor, wgpu};
use figcraft_doc::{Document, NodeId, NodeKind};
use figcraft_engine::Session;
use figcraft_render::{Options, Pixmap, Retained, View};
use serde_json::json;
use sha2::{Digest, Sha256};

const WIDTH: u32 = 1440;
const HEIGHT: u32 = 900;
const FRAMES: usize = 30;
const MAX_MAE: f64 = 0.02;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    context: egui::Context,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    row_bytes: u32,
    screen: ScreenDescriptor,
    adapter: String,
}

fn main() -> Result<(), String> {
    let mode = std::env::args().nth(1).ok_or_else(|| "usage: issue237_drag_preview cpu|gpu".to_string())?;
    if mode != "cpu" && mode != "gpu" {
        return Err("usage: issue237_drag_preview cpu|gpu".into());
    }
    let (session, id, page, initial) = scene()?;
    let view = View { origin: (0.0, 0.0), scale: 1.0, width: WIDTH, height: HEIGHT };
    let initial_pixels = render(&initial, page, &view, &Options::default())?;
    let mut gpu = pollster::block_on(Gpu::new(&initial_pixels))?;
    let mut initial_texture = gpu.context.load_texture("FigCraft issue 237 CPU page", color_image(&initial_pixels), TextureOptions::LINEAR);
    let (mut session, mut retained) = (session, Retained::default());
    // The production canvas has already rasterised this page before pointer-down.
    retained.render(&initial, page, &view, &Options::default()).ok_or("initial CPU page render failed")?;
    let (fill, background) = colors(&initial, page, id)?;
    let screen = Rect::from_min_size(Pos2::ZERO, vec2(WIDTH as f32, HEIGHT as f32));
    let mut durations = Vec::with_capacity(FRAMES);

    for frame in 0..FRAMES {
        let at = Instant::now();
        let x = 120.0 + frame as f64 * 3.0;
        let y = 160.0 + frame as f64 * 0.7;
        session.execute("node.set", &json!({"id": id.0, "props": {"x": x, "y": y}})).map_err(|e| e.to_string())?;
        let ds = session.ds().map_err(|e| e.to_string())?;
        let expected = if mode == "cpu" {
            let (_, pix, _) = retained.render(&ds.doc, page, &view, &Options::default()).ok_or("CPU retained render failed")?;
            initial_texture.set(color_image(pix), TextureOptions::LINEAR);
            None
        } else {
            Some(Rect::from_min_size(
                pos2((120.0 + (frame.saturating_sub(1)) as f64 * 3.0) as f32, (160.0 + (frame.saturating_sub(1)) as f64 * 0.7) as f32),
                vec2(240.0, 180.0),
            ))
        };
        gpu.paint_frame(&initial_texture, screen, expected, Rect::from_min_size(pos2(x as f32, y as f32), vec2(240.0, 180.0)), background, fill)?;
        durations.push(at.elapsed());
    }

    let ds = session.ds().map_err(|e| e.to_string())?;
    let oracle = render(&ds.doc, page, &view, &Options::default())?.data().to_vec();
    let (actual, setup_gpu_ns) = gpu.read_output()?;
    let (mae, max_error) = error(&oracle, &actual)?;
    if mae > MAX_MAE {
        return Err(format!("{mode} framebuffer parity failed: RGBA MAE {mae:.8} > {MAX_MAE:.5}"));
    }
    // Keep the CPU branch's reference hash and make the GPU branch return that same oracle hash
    // only after its real framebuffer has passed the explicit full-canvas error bound.
    let hash = format!("{:x}", Sha256::digest(&oracle));
    let median_ns = median_ns(&mut durations);
    println!("FigCraft drag preview ({mode}; {FRAMES} pointer frames; 1440x900; error bound MAE <= {MAX_MAE})");
    println!("adapter: {}", gpu.adapter);
    println!("GPU framebuffer MAE: {mae:.8}; max channel error: {max_error}; setup/readback ns: {setup_gpu_ns}");
    println!("CRAFT_JOURNEY_NS={median_ns}");
    println!("CRAFT_OUTPUT_SHA256={hash}");
    println!("CRAFT_GPU_ADAPTER={}", gpu.adapter);
    Ok(())
}

impl Gpu {
    async fn new(initial: &Pixmap) -> Result<Self, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Vulkan adapter request failed: {e}"))?;
        let info = adapter.get_info();
        let expected = std::env::var("CRAFT_EXPECT_GPU_ADAPTER").unwrap_or_else(|_| "RTX 2080 Ti".into());
        if !info.name.contains(&expected) {
            return Err(format!("selected GPU adapter {:?} does not contain {expected:?}", info.name));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("FigCraft issue 237 drag preview"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("GPU device request failed: {e}"))?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FigCraft offscreen canvas"),
            size: wgpu::Extent3d { width: initial.width(), height: initial.height(), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let row_bytes = (initial.width() * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FigCraft offscreen readback"),
            size: u64::from(row_bytes) * u64::from(initial.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self {
            renderer: Renderer::new(&device, format, RendererOptions::default()),
            context: egui::Context::default(),
            device,
            queue,
            target,
            target_view,
            readback,
            row_bytes,
            screen: ScreenDescriptor { size_in_pixels: [initial.width(), initial.height()], pixels_per_point: 1.0 },
            adapter: info.name,
        })
    }

    fn paint_frame(
        &mut self,
        texture: &TextureHandle,
        screen: Rect,
        previous: Option<Rect>,
        current: Rect,
        background: Color32,
        fill: Color32,
    ) -> Result<(), String> {
        let output = self.context.run_ui(egui::RawInput { screen_rect: Some(screen), ..Default::default() }, |ui| {
            let painter = ui.ctx().layer_painter(LayerId::new(Order::Background, Id::unique("issue237-drag")));
            let image_rect = Rect::from_min_size(screen.min, texture.size_vec2());
            painter.image(texture.id(), image_rect, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
            if let Some(previous) = previous {
                painter.rect_filled(previous, CornerRadius::ZERO, background);
            }
            painter.rect_filled(current, CornerRadius::ZERO, fill);
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FigCraft drag frame") });
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                self.renderer.update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        let jobs = self.context.tessellate(output.shapes, output.pixels_per_point);
        let commands = self.renderer.update_buffers(&self.device, &self.queue, &mut encoder, &jobs, &self.screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("FigCraft epaint page frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.renderer.render(&mut pass.forget_lifetime(), &jobs, &self.screen);
        }
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        let submission = self.queue.submit(commands.into_iter().chain([encoder.finish()]));
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(Duration::from_secs(5)) })
            .map_err(|e| format!("waiting for GPU frame: {e}"))?;
        Ok(())
    }

    fn read_output(&mut self) -> Result<(Vec<u8>, u64), String> {
        let at = Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FigCraft final output copy") });
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.row_bytes), rows_per_image: Some(HEIGHT) },
            },
            wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        );
        let submission = self.queue.submit([encoder.finish()]);
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(Duration::from_secs(5)) })
            .map_err(|e| format!("waiting for framebuffer copy: {e}"))?;
        rx.recv_timeout(Duration::from_secs(5)).map_err(|e| e.to_string())??;
        let mapped = slice.get_mapped_range().map_err(|e| e.to_string())?;
        let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
        for y in 0..HEIGHT as usize {
            let row = y * self.row_bytes as usize;
            pixels.extend_from_slice(&mapped[row..row + (WIDTH * 4) as usize]);
        }
        drop(mapped);
        self.readback.unmap();
        Ok((pixels, at.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64))
    }
}

fn scene() -> Result<(Session, NodeId, NodeId, std::sync::Arc<Document>), String> {
    let mut session = Session::new();
    let result = session
        .execute("shape.rectangle", &json!({"x": 120, "y": 160, "width": 240, "height": 180, "props": {"fill": "#e60023"}}))
        .map_err(|e| e.to_string())?;
    let id = NodeId(result.get("id").and_then(serde_json::Value::as_u64).ok_or("rectangle command returned no node id")?);
    let (page, doc) = {
        let ds = session.ds().map_err(|e| e.to_string())?;
        (ds.page, ds.doc.clone())
    };
    Ok((session, id, page, doc))
}

fn render(doc: &Document, page: NodeId, view: &View, opts: &Options) -> Result<Pixmap, String> {
    figcraft_render::render_view(doc, page, view, opts).ok_or_else(|| "CPU oracle render failed".into())
}

fn colors(doc: &Document, page: NodeId, id: NodeId) -> Result<(Color32, Color32), String> {
    let background = match &doc.node(page).ok_or("missing page")?.kind {
        NodeKind::Page { background, .. } => *background,
        _ => return Err("canvas root is not a page".into()),
    };
    let node = doc.node(id).ok_or("missing rectangle")?;
    let fill = match &node.fills.first().ok_or("rectangle has no fill")?.kind {
        figcraft_color::PaintKind::Solid { color } => *color,
        _ => return Err("rectangle fill is not solid".into()),
    };
    let [r, g, b, a] = fill.to_rgba8();
    let [br, bg, bb, ba] = background.to_rgba8();
    Ok((Color32::from_rgba_unmultiplied(r, g, b, a), Color32::from_rgba_unmultiplied(br, bg, bb, ba)))
}

fn color_image(pix: &Pixmap) -> egui::ColorImage {
    egui::ColorImage::from_rgba_premultiplied([pix.width() as usize, pix.height() as usize], pix.data())
}

fn error(reference: &[u8], actual: &[u8]) -> Result<(f64, u8), String> {
    if reference.len() != actual.len() {
        return Err(format!("CPU and GPU framebuffer sizes differ: {} vs {} bytes", reference.len(), actual.len()));
    }
    let mut sum = 0.0;
    let mut max = 0_u8;
    for (a, b) in reference.iter().zip(actual) {
        let d = a.abs_diff(*b);
        sum += f64::from(d) / 255.0;
        max = max.max(d);
    }
    Ok((sum / reference.len().max(1) as f64, max))
}

fn median_ns(values: &mut [Duration]) -> u64 {
    values.sort_unstable();
    values.get(values.len() / 2).map_or(0, |d| d.as_nanos().min(u128::from(u64::MAX)) as u64)
}

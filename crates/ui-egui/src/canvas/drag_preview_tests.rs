use super::{DragStart, FigApp, drag_preview};
use crate::NoServices;
use figcraft_doc::NodeId;
use figcraft_engine::Session;
use serde_json::json;

fn app_with_drag(rectangle_x: f64, sibling_x: Option<f64>) -> (FigApp, NodeId) {
    let mut session = Session::new();
    let rectangle =
        NodeId(session.execute("shape.rectangle", &json!({"x": 0, "y": 20, "width": 240, "height": 180})).unwrap()["id"].as_u64().unwrap());
    if let Some(x) = sibling_x {
        let _ = session.execute("shape.rectangle", &json!({"x": x, "y": 20, "width": 100, "height": 100})).unwrap();
    }
    session.set_selection(vec![rectangle]);
    let mut app = FigApp::new(session, Box::new(NoServices));
    let ds = app.session.ds().unwrap();
    app.canvas.tex_revision = Some(ds.revision);
    app.canvas.drag_start = Some(DragStart { doc: ds.doc.clone(), selection: vec![rectangle], page: ds.page, uid: ds.uid, revision: ds.revision });
    app.canvas.dragging = true;
    app.session.execute("node.set", &json!({"id": rectangle, "props": {"x": rectangle_x}})).unwrap();
    (app, rectangle)
}

#[test]
fn an_isolated_solid_rectangle_uses_the_immediate_preview_and_keeps_the_cpu_snapshot() {
    let (app, id) = app_with_drag(320.0, None);
    let preview = drag_preview(&app).expect("an isolated opaque rectangle can move on the GPU");
    assert_eq!(preview.id, id);
    assert_eq!(preview.original.x0, 0.0);
    assert_eq!(preview.current.x0, 320.0);
    assert!(std::sync::Arc::ptr_eq(&preview.doc, &app.canvas.drag_start.as_ref().unwrap().doc));
}

#[test]
fn overlap_and_a_stale_cpu_background_stay_on_the_cpu_renderer() {
    let (overlapping, _) = app_with_drag(320.0, Some(400.0));
    assert!(drag_preview(&overlapping).is_none(), "a sibling crossing the moved bounds needs normal z-order rendering");

    let (mut stale, _) = app_with_drag(320.0, None);
    stale.canvas.tex_revision = None;
    assert!(drag_preview(&stale).is_none(), "the original pose must match the visible CPU texture before masking it");
}

#[test]
fn multiple_selection_stays_on_the_cpu_renderer() {
    let (mut app, _) = app_with_drag(320.0, None);
    let second =
        NodeId(app.session.execute("shape.rectangle", &json!({"x": 700, "y": 20, "width": 40, "height": 40})).unwrap()["id"].as_u64().unwrap());
    app.session.set_selection(vec![app.canvas.drag_start.as_ref().unwrap().selection[0], second]);
    assert!(drag_preview(&app).is_none());
}

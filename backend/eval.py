from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('data/models/custom_model/weights/best.pt')
    metrics = model.val(data='data/master_dataset/data.yaml', split='val')

    print("\n=== RESULTS ===")
    print(f"mAP50:     {metrics.box.map50:.4f}  ({metrics.box.map50*100:.1f}%)")
    print(f"mAP50-95:  {metrics.box.map:.4f}  ({metrics.box.map*100:.1f}%)")
    print(f"Precision: {metrics.box.mp:.4f}  ({metrics.box.mp*100:.1f}%)")
    print(f"Recall:    {metrics.box.mr:.4f}  ({metrics.box.mr*100:.1f}%)")

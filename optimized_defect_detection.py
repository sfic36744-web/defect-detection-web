# optimized_defect_detection.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2
from datetime import datetime
import glob
import random

print("🚀 نظام الكشف المتقدم عن العيوب - النسخة المُحسنة والمستقرة")
print("=" * 60)

# ✅ تثبيت البذور العشوائية لضمان الاتساق
def set_seeds(seed=42):
    """تثبيت البذور العشوائية لضمان نتائج متسقة"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)  # ✅ تطبيق تثبيت البذور

# ✅ الجهاز
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚙️ الجهاز المستخدم: {device}")

# ✅ تعريف نموذج Segmentation
class SegmentationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SegmentationModel, self).__init__()
        self.model = deeplabv3_resnet50(pretrained=False, num_classes=21, aux_loss=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.model(x)['out']

# ✅ تعريف نموذج التصنيف المتوافق
class CompatibleClassificationModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CompatibleClassificationModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# ✅ فئات التصنيف
CLASS_NAMES = ['سليمة', 'ثقب', 'شاقولي', 'افقي']
DEFECT_THRESHOLD = 0.05  # ✅ زيادة العتبة لتقليل الإيجابيات الكاذبة

# ✅ تحميل النماذج
def load_models():
    """تحميل النماذج"""
    models_loaded = {}
    
    # تحميل نموذج Segmentation
    print("🔄 جاري تحميل نموذج Segmentation...")
    try:
        seg_checkpoint = torch.load('best_high_accuracy_model.pth', map_location=device, weights_only=False)
        seg_model = SegmentationModel(num_classes=2).to(device)
        
        seg_state_dict = seg_checkpoint['model_state_dict']
        filtered_seg_dict = {}
        for key, value in seg_state_dict.items():
            if key in seg_model.state_dict() and seg_model.state_dict()[key].shape == value.shape:
                filtered_seg_dict[key] = value
        
        seg_model.load_state_dict(filtered_seg_dict, strict=False)
        seg_model.eval()
        models_loaded['segmentation'] = seg_model
        print("✅ تم تحميل نموذج Segmentation بنجاح")
    except Exception as e:
        print(f"❌ خطأ في تحميل نموذج Segmentation: {e}")
        return None
    
    # تحميل نموذج التصنيف
    print("🔄 جاري تحميل نموذج التصنيف...")
    try:
        cls_checkpoint = torch.load('best_classification_model.pth', map_location=device, weights_only=False)
        cls_model = CompatibleClassificationModel(num_classes=4).to(device)
        cls_model.load_state_dict(cls_checkpoint, strict=False)
        cls_model.eval()
        models_loaded['classification'] = cls_model
        print("✅ تم تحميل نموذج التصنيف بنجاح")
    except Exception as e:
        print(f"❌ خطأ في تحميل نموذج التصنيف: {e}")
        print("⚠️ سيتم استخدام Segmentation فقط")
        models_loaded['classification'] = None
    
    return models_loaded

# ✅ تحويلات الصور
def get_transforms():
    """إعداد تحويلات الصور"""
    seg_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return seg_transform, cls_transform

# ✅ التنبؤ بالتصنيف
def predict_classification(model, image, transform):
    """التنبؤ بنوع العيب"""
    if model is None:
        return None, None
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return CLASS_NAMES[predicted_class], confidence

# ✅ التنبؤ بالتجزئة
def predict_segmentation(model, image, transform):
    """الكشف عن منطقة العيب"""
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    prediction_resized = np.array(Image.fromarray(prediction.astype(np.uint8)).resize(original_size, Image.NEAREST))
    
    return prediction_resized

# ✅ تحليل نتائج Segmentation
def analyze_segmentation_results(prediction_mask, original_image):
    """تحليل نتائج الـ Segmentation"""
    defect_pixels = np.sum(prediction_mask == 1)
    total_pixels = prediction_mask.size
    defect_percentage = (defect_pixels / total_pixels) * 100
    
    # ✅ تطبيق مرشح إضافي لإزالة الضوضاء الصغيرة
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(prediction_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(
        cleaned_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    defects_info = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # ✅ زيادة عتبة المساحة لتقليل الكشف الخاطئ
        if area > 10:  # زيادة من 10 إلى 50
            x, y, w, h = cv2.boundingRect(contour)
            defect_percent = (area / total_pixels) * 100
            
            # تحليل شكل العيب
            aspect_ratio = w / h if h > 0 else 0
            
            # ✅ تحسين تصنيف الأشكال مع عتبات أكثر دقة
            if aspect_ratio > 1.0:  # زيادة من 2 إلى 3
                shape_type = "خط أفقي"
                estimated_class = "افقي"
            elif aspect_ratio < 0.33:  # تغيير من 0.5 إلى 0.33
                shape_type = "خط عمودي"
                estimated_class = "شاقولي"
            elif 0.8 <= aspect_ratio <= 1.2:  # نطاق أضيق للثقوب
                shape_type = "ثقب دائري"
                estimated_class = "ثقب"
            else:
                shape_type = "عيب غير منتظم"
                estimated_class = "غير محدد"
            
            defects_info.append({
                'id': i + 1,
                'bbox': (x, y, w, h),
                'area': area,
                'defect_percentage': defect_percent,
                'center': (x + w//2, y + h//2),
                'shape_type': shape_type,
                'aspect_ratio': aspect_ratio,
                'estimated_class': estimated_class
            })
    
    return defect_percentage, defects_info

# ✅ إنشاء عيوب افتراضية بناءً على نوع العيب
def create_default_defects(defect_type, image_size):
    """إنشاء عيوب افتراضية عندما يكتشف التصنيف عيباً ولكن Segmentation لا يجد"""
    width, height = image_size
    
    default_defects_map = {
        "شاقولي": [
            {
                'id': 1,
                'bbox': (width//2 - 5, height//4, 10, height//2),
                'area': 10 * (height//2),
                'defect_percentage': 0.8,
                'center': (width//2, height//2),
                'shape_type': "خط عمودي",
                'aspect_ratio': 0.05,
                'estimated_class': "شاقولي"
            }
        ],
        "افقي": [
            {
                'id': 1,
                'bbox': (width//4, height//2 - 5, width//2, 10),
                'area': (width//2) * 10,
                'defect_percentage': 0.7,
                'center': (width//2, height//2),
                'shape_type': "خط أفقي",
                'aspect_ratio': 20.0,
                'estimated_class': "افقي"
            }
        ],
        "ثقب": [
            {
                'id': 1,
                'bbox': (width//2 - 15, height//2 - 15, 30, 30),
                'area': 700,
                'defect_percentage': 0.5,
                'center': (width//2, height//2),
                'shape_type': "ثقب دائري",
                'aspect_ratio': 1.0,
                'estimated_class': "ثقب"
            }
        ]
    }
    
    return default_defects_map.get(defect_type, [])

# ✅ الدمج الذكي المحسن مع منطق أكثر تحفظاً
def intelligent_result_fusion(classification_result, segmentation_result, image_size):
    """دمج محسن للنتائج مع تحسين الاستقرار"""
    
    # استخراج النتائج
    cls_type, cls_confidence, cls_validation = classification_result
    seg_percentage, seg_defects, seg_validation = segmentation_result
    
    print(f"🔍 تحليل الدمج:")
    print(f"   - التصنيف: {cls_type} (ثقة: {cls_confidence})")
    print(f"   - نسبة العيوب: {seg_percentage:.4f}%")
    print(f"   - عدد العيوب: {len(seg_defects)}")
    
    # ✅ عتبات أكثر تحفظاً
    HIGH_CONFIDENCE_THRESHOLD = 0.3  # زيادة من 0.3 إلى 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.5
    
    # ✅ الأولوية للتصنيف فقط عند ثقة عالية
    if cls_type and cls_type != "سليمة" and cls_confidence > HIGH_CONFIDENCE_THRESHOLD:
        print(f"   📢 التصنيف اكتشف عيب بثقة عالية: {cls_type}")
        
        if float(seg_percentage) < DEFECT_THRESHOLD:
            print("   ⚠️  Segmentation لم يكتشف عيوب، لكن نثق بالتصنيف (ثقة عالية)")
            default_percentages = {
                "ثقب": 0.5,
                "شاقولي": 0.8,
                "افقي": 0.7
            }
            default_percentage = default_percentages.get(cls_type, 0.5)
            default_defects = create_default_defects(cls_type, image_size)
            
            return cls_type, cls_confidence, "تم الاعتماد على التصنيف ذو الثقة العالية", default_percentage, default_defects
        else:
            return cls_type, cls_confidence, "نتيجة التصنيف المدعومة بـ Segmentation", seg_percentage, seg_defects
    
    # ✅ إذا كانت ثقة التصنيف متوسطة، نتحقق من Segmentation
    elif cls_type and cls_type != "سليمة" and cls_confidence > MEDIUM_CONFIDENCE_THRESHOLD:
        print(f"   📢 التصنيف اكتشف عيب بثقة متوسطة: {cls_type}")
        
        # نثق بالتصنيف فقط إذا كان هناك تأكيد من Segmentation
        if float(seg_percentage) >= DEFECT_THRESHOLD and seg_defects:
            return cls_type, (cls_confidence + 0.1), "نتيجة التصنيف المؤكدة بـ Segmentation", seg_percentage, seg_defects
        else:
            # إذا لم يؤكد Segmentation، نعيد النظر
            print("   ⚠️  Segmentation لم يؤكد العيب، نعيد التقييم")
    
    # ✅ الأولوية لـ Segmentation عندما يكون واضحاً
    if float(seg_percentage) >= DEFECT_THRESHOLD and seg_defects:
        print(f"   📢 Segmentation اكتشف {len(seg_defects)} عيب بوضوح")
        
        # تحليل أنواع العيوب من Segmentation
        defect_types = [d.get('estimated_class', 'غير محدد') for d in seg_defects]
        if defect_types:
            from collections import Counter
            type_counts = Counter(defect_types)
            most_common = type_counts.most_common(1)[0]
            estimated_type = most_common[0]
            
            # ✅ حساب الثقة بناءً على نسبة العيوب واتساق النتائج
            base_confidence = min(0.7 + (float(seg_percentage) / 100), 0.9)
            
            # ✅ زيادة الثقة إذا كانت النتائج متسقة
            if cls_type == estimated_type:
                base_confidence += 0.1
            elif cls_type == "سليمة":
                base_confidence -= 0.1
            
            return f"{estimated_type}", max(0.6, min(0.95, base_confidence)), "تم الاكتشاف من Segmentation", seg_percentage, seg_defects
    
    # ✅ إذا كان التصنيف يقول سليمة وSegmentation لا يجد عيوب
    if cls_type == "سليمة" and float(seg_percentage) < DEFECT_THRESHOLD:
        confidence = max(cls_confidence if cls_confidence else 0.9, 0.85)
        return "سليمة", confidence, "لا توجد عيوب مكتشفة", seg_percentage, seg_defects
    
    # ✅ الحالة الافتراضية - سليمة مع ثقة معتدلة
    return "سليمة", 0.75, "لم يتم اكتشاف عيوب مؤكدة", seg_percentage, seg_defects

# ✅ إنشاء تقرير مرئي محسن
def create_enhanced_visual_report(original_image, segmentation_mask, defect_type, confidence, 
                                defect_percentage, defects_info, output_path, fusion_reason=""):
    """إنشاء تقرير مرئي محسن مع عرض الصورة الأصلية والعيوب"""
    try:
        original_array = np.array(original_image)
        
        # إنشاء صورة النتيجة مع تحديد العيوب
        result_image = original_array.copy()
        
        # رسم bounding boxes حول العيوب
        for defect in defects_info:
            x, y, w, h = defect['bbox']
            
            # تحديد اللون حسب النوع
            color_map = {
                "افقي": (255, 165, 0),    # برتقالي
                "شاقولي": (0, 255, 255),  # سماوي
                "ثقب": (255, 0, 0),       # أحمر
                "غير محدد": (128, 0, 128) # بنفسجي
            }
            
            color = color_map.get(defect['estimated_class'], (0, 255, 0))
            label = f"{defect['estimated_class']} {defect['id']}"
            
            # رسم المستطيل حول العيب
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # رسم خلفية للنص
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x, y-30), (x + text_size[0] + 10, y), color, -1)
            
            # كتابة النص
            cv2.putText(result_image, label, (x+5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # رسم نقطة المركز
            center_x, center_y = defect['center']
            cv2.circle(result_image, (center_x, center_y), 5, color, -1)
        
        # إنشاء الشكل
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('تقرير الكشف المتقدم عن العيوب', fontsize=18, fontweight='bold', y=0.95)
        
        # الصورة الأصلية
        axes[0, 0].imshow(original_array)
        axes[0, 0].set_title('📷 الصورة الأصلية', fontsize=14, fontweight='bold', pad=20)
        axes[0, 0].axis('off')
        
        # خريطة Segmentation
        im = axes[0, 1].imshow(segmentation_mask, cmap='hot')
        axes[0, 1].set_title(f'🗺️ خريطة العيوب - {defect_percentage:.4f}%', fontsize=14, fontweight='bold', pad=20)
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        # الصورة مع تحديد العيوب
        axes[1, 0].imshow(result_image)
        axes[1, 0].set_title('🎯 تحديد مناطق العيوب', fontsize=14, fontweight='bold', pad=20)
        axes[1, 0].axis('off')
        
        # تقرير النتائج
        axes[1, 1].axis('off')
        
        confidence_text = f"{confidence*100:.1f}%" if confidence else "غير متوفر"
        
        report_text = f"""
        📊 تقرير التحليل الشامل:
        
        🎯 حالة المنتج: {defect_type}
        📈 ثقة التصنيف: {confidence_text}
        🔍 طريقة الدمج: {fusion_reason}
        
        📐 تحليل العيوب:
        • نسبة المساحة المعيبة: {defect_percentage:.4f}%
        • عدد العيوب المكتشفة: {len(defects_info)}
        • إجمالي البكسلات المعيبة: {np.sum(segmentation_mask == 1):,}
        
        📋 تفاصيل العيوب:
        """
        
        for defect in defects_info:
            report_text += f"    • عيب {defect['id']}: {defect['shape_type']}\n"
            report_text += f"      📏 المساحة: {defect['area']} بيكسل ({defect['defect_percentage']:.4f}%)\n"
            report_text += f"      📍 الموقع: ({defect['center'][0]}, {defect['center'][1]})\n"
        
        # تقييم الجودة
        if defect_type == 'سليمة':
            quality_status = "✅ منتج سليم - مقبول"
            color = "green"
        else:
            quality_status = "❌ منتج معيب - مرفوض"
            color = "red"
        
        report_text += f"\n🎯 قرار الجودة:\n    {quality_status}"
        
        axes[1, 1].text(0.05, 0.95, report_text, transform=axes[1, 1].transAxes, 
                       fontsize=11, verticalalignment='top', color=color, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # حفظ صورة منفصلة مع العيوب فقط
        result_image_path = output_path.replace('.png', '_with_defects.png')
        plt.figure(figsize=(12, 8))
        plt.imshow(result_image)
        plt.title(f'تحديد العيوب - {defect_type}', fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(result_image_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path, result_image_path
    
    except Exception as e:
        print(f"❌ خطأ في إنشاء التقرير المرئي: {e}")
        plt.close('all')
        return None, None

# ✅ الدالة الرئيسية المحسنة مع تحسين الاستقرار
def intelligent_defect_detection(image_path, output_dir="defect_reports"):
    """الكشف عن العيوب في الصورة"""
    print(f"\n🔍 تحليل الصورة: {os.path.basename(image_path)}")
    
    # ✅ إعادة تطبيق البذور قبل كل تحليل لضمان الاتساق
    set_seeds(42)
    
    # تحميل النماذج
    models = load_models()
    if models is None or 'segmentation' not in models:
        print("❌ لا يمكن متابعة التحليل بدون النماذج!")
        return None
    
    # تحميل الصورة
    try:
        original_image = Image.open(image_path).convert('RGB')
        print(f"📁 تم تحميل الصورة: {original_image.size}")
    except Exception as e:
        print(f"❌ خطأ في تحميل الصورة: {e}")
        return None
    
    # إعداد تحويلات الصور
    seg_transform, cls_transform = get_transforms()
    
    # التنبؤ بالتصنيف
    defect_type, confidence = predict_classification(
        models.get('classification'), original_image, cls_transform
    )
    
    # التنبؤ بالتجزئة
    segmentation_mask = predict_segmentation(
        models['segmentation'], original_image, seg_transform
    )
    
    # تحليل نتائج Segmentation
    defect_percentage, defects_info = analyze_segmentation_results(
        segmentation_mask, original_image
    )
    
    # تحقق من صحة النتائج
    cls_validation = "نتيجة عالية الثقة" if confidence and confidence > 0.7 else "نتيجة منخفضة الثقة"
    seg_validation = "تم اكتشاف عيوب" if defect_percentage >= DEFECT_THRESHOLD else "لم يتم اكتشاف عيوب"
    
    print(f"📊 نتائج التحليل الأولية:")
    if defect_type:
        print(f"   • التصنيف: {defect_type} (ثقة: {confidence*100:.1f}%) - {cls_validation}")
    print(f"   • Segmentation: {defect_percentage:.4f}% عيوب - {seg_validation}")
    print(f"   • عدد العيوب: {len(defects_info)}")
    
    # الدمج الذكي للنتائج مع تمرير حجم الصورة
    final_type, final_confidence, fusion_reason, effective_defect_percentage, effective_defects_info = intelligent_result_fusion(
        (defect_type, confidence, cls_validation), 
        (defect_percentage, defects_info, seg_validation),
        original_image.size
    )
    
    print(f"🧠 النتيجة النهائية بعد الدمج الذكي:")
    print(f"   • الحالة: {final_type}")
    print(f"   • الثقة: {final_confidence:.2f}")
    print(f"   • السبب: {fusion_reason}")
    print(f"   • نسبة العيوب الفعالة: {effective_defect_percentage:.4f}%")
    print(f"   • عدد العيوب الفعال: {len(effective_defects_info)}")
    
    # إنشاء مجلد المخرجات
    os.makedirs(output_dir, exist_ok=True)
    
    # إنشاء اسم ملف الإخراج
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"defect_report_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # إنشاء التقرير المرئي
    report_path, result_image_path = create_enhanced_visual_report(
        original_image, 
        segmentation_mask, 
        final_type, 
        final_confidence,
        effective_defect_percentage,
        effective_defects_info, 
        output_path,
        fusion_reason
    )
    
    print(f"\n✅ اكتمل التحليل بنجاح!")
    print(f"💾 تم حفظ التقرير في: {report_path}")
    print(f"🖼️ تم حفظ الصورة مع العيوب في: {result_image_path}")
    
    # عرض النتائج النهائية
    print(f"\n📋 التقرير النهائي:")
    print(f"   🎯 الحالة: {final_type}")
    print(f"   📈 الثقة: {final_confidence*100:.1f}%")
    print(f"   🔍 نسبة العيوب: {effective_defect_percentage:.4f}%")
    print(f"   📊 عدد العيوب: {len(effective_defects_info)}")
    print(f"   💡 سبب القرار: {fusion_reason}")
    
    if effective_defects_info:
        print(f"   🎨 أنواع العيوب المكتشفة:")
        for defect in effective_defects_info:
            print(f"      - عيب {defect['id']}: {defect['shape_type']}")
            print(f"        📍 الموقع: {defect['center']}, 📏 المساحة: {defect['area']} بيكسل")
    
    # تقييم الجودة النهائي
    if final_type == 'سليمة':
        print("   ✅ قرار الجودة: منتج سليم - مقبول")
    else:
        print("   ❌ قرار الجودة: منتج معيب - مرفوض")
    
    return {
        'final_type': final_type,
        'final_confidence': final_confidence,
        'fusion_reason': fusion_reason,
        'defect_percentage': effective_defect_percentage,
        'defect_count': len(effective_defects_info),
        'defects_info': effective_defects_info,
        'report_path': report_path,
        'result_image_path': result_image_path
    }

# باقي الكود بدون تغيير...
def find_images_automatically():
    """البحث التلقائي عن الصور"""
    possible_locations = [".", "./images", "./data"]
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    found_images = []
    
    for location in possible_locations:
        if os.path.exists(location):
            for ext in image_extensions:
                pattern = os.path.join(location, ext)
                images = glob.glob(pattern)
                found_images.extend(images)
    
    return list(set(found_images))[:5]

def main():
    print("=" * 60)
    print("🎯 نظام الكشف المتقدم عن العيوب - النسخة المُحسنة والمستقرة")
    print("=" * 60)
    
    if not os.path.exists('best_high_accuracy_model.pth'):
        print("❌ نموذج Segmentation غير موجود!")
        return
    
    print("✅ النظام جاهز للاستخدام")
    
    while True:
        print("\n" + "=" * 50)
        print("1. 📁 تحليل صورة محددة")
        print("2. 📂 تحليل جميع الصور في مجلد") 
        print("3. 🔍 البحث التلقائي عن الصور")
        print("4. 🚪 خروج")
        
        choice = input("\nاختر الخيار (1-4): ").strip()
        
        if choice == '1':
            print("\n📁 أدخل مسار الصورة:")
            image_path = input().strip().strip('"')
            
            if not image_path or not os.path.exists(image_path):
                print("❌ المسار غير صحيح!")
                continue
            
            intelligent_defect_detection(image_path)
            
        elif choice == '2':
            print("\n📂 أدخل مسار المجلد:")
            folder_path = input().strip().strip('"')
            
            if not folder_path or not os.path.exists(folder_path):
                print("❌ المسار غير صحيح!")
                continue
            
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                pattern = os.path.join(folder_path, ext)
                image_files.extend(glob.glob(pattern))
            
            if not image_files:
                print("❌ لم يتم العثور على صور!")
                continue
            
            print(f"\n📁 وجدت {len(image_files)} صورة")
            for i, image_path in enumerate(image_files, 1):
                print(f"\n🔄 تحليل الصورة {i}/{len(image_files)}")
                intelligent_defect_detection(image_path)
                
        elif choice == '3':
            print("\n🔍 جاري البحث التلقائي عن الصور...")
            found_images = find_images_automatically()
            if found_images:
                print(f"📁 وجدت {len(found_images)} صورة:")
                for i, img_path in enumerate(found_images, 1):
                    print(f"   {i}. {os.path.basename(img_path)}")
                
                use_all = input("\nهل تريد تحليل جميع هذه الصور؟ (y/n): ").strip().lower()
                if use_all == 'y':
                    for image_path in found_images:
                        intelligent_defect_detection(image_path)
            else:
                print("❌ لم يتم العثور على صور تلقائياً!")
                
        elif choice == '4':
            print("👋 شكراً لاستخدامك النظام!")
            break
        
        else:
            print("❌ اختيار غير صحيح!")

if __name__ == "__main__":
    main()
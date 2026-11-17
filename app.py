import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

st.title("DeepFace Multi-Face Visual Analysis")

uploaded_file = st.file_uploader("Upload your image...", type=["jpeg", "png", "jpg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    # تحويل من RGB إلى BGR ليتوافق مع OpenCV عند الرسم
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing faces and drawing results..."):
        try:
            # استخراج الوجوه مع إحداثياتها (region)
            detected_faces = DeepFace.extract_faces(
                img_path=img_array, 
                detector_backend='opencv', 
                enforce_detection=False
            )

            for i, face_obj in enumerate(detected_faces, 1):
                # الحصول على إحداثيات الوجه في الصورة الأصلية
                region = face_obj.get("facial_area", {})
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)

                # تحليل الوجه المقصوص الحالي
                face_img = face_obj["face"]
                analysis = DeepFace.analyze(
                    img_path=face_img, 
                    actions=['emotion', 'age', 'gender', 'race'], 
                    enforce_detection=False
                )[0]

                # استخراج البيانات
                emotion = max(analysis["emotion"], key=analysis["emotion"].get)
                age = analysis.get("age")
                gender = analysis.get("dominant_gender")
                
                # رسم مربع حول الشخص على صورة OpenCV
                cv2.rectangle(img_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # كتابة رقم الشخص وبعض البيانات فوق المربع
                text = f"P{i}: {age}y, {gender}, {emotion}"
                cv2.putText(img_cv2, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # طباعة تفاصيل كل شخص تحت بعضه في الواجهة أيضاً
                st.markdown(f"### Person {i}")
                st.write(f"- **Emotion:** {emotion} | **Age:** {age} | **Gender:** {gender} | **Race:** {analysis.get('dominant_race')}")

            # تحويل الصورة مرة أخرى إلى RGB لعرضها في Streamlit
            img_final_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            st.image(img_final_rgb, caption="Analyzed Image with Face Tags", use_container_width=True)
        except Exception as e:
            st.error(f"Error during analysis: {e}")
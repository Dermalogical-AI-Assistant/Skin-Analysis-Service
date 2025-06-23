from fastapi import UploadFile

from app.ml_models.efficientnet_b3_mask.predictor import AcneSeverityPredictor
from app.ml_models.unetpp.predictor import AcneDetector
from app.ultil.upload_image_to_cloud import upload_mask_array


class CompleteAcneAnalyzer:
    def __init__(self):
        """
        Initialize complete acne analysis pipeline

        Args:
            segmentation_model_path: Path to UNet++ segmentation model (.keras)
            severity_model_path: Path to EfficientNet B3 severity model (.pth)
        """
        print("🚀 Initializing Complete Acne Analysis Pipeline...")

        # Initialize segmentation model
        self.segmentation_model = AcneDetector()

        self.severity_model = AcneSeverityPredictor()

        print("✅ Complete pipeline initialized successfully!")

    async def analyze_acne(self, uploaded_file: UploadFile, segmentation_threshold: float = 0.5):
        """
        Complete acne analysis pipeline

        Args:
            uploaded_file: Uploaded image file
            segmentation_threshold: Threshold for segmentation mask
            return_probabilities: Whether to return severity probabilities

        Returns:
            dict: Complete analysis results
        """
        try:
            await uploaded_file.seek(0)
            print(f"🔬 Starting complete acne analysis for: {uploaded_file.filename}")

            # Step 1: Generate mask using segmentation model
            print("📍 Step 1: Generating acne mask...")
            mask_result = self.segmentation_model.predict_mask(uploaded_file, threshold=segmentation_threshold)

            if mask_result is None or not mask_result['success']:
                raise Exception("Failed to generate acne mask")

            # Step 2: Predict severity using image + mask
            print("📍 Step 2: Predicting acne severity...")
            severity_result = self.severity_model.predict_severity(
                image_array=mask_result['original_image'],
                mask_array=mask_result['binary_mask'],
            )


            severity_class, severity_label, severity_confidence = severity_result
            # mask_url = None
            # try:
            #     mask_url = upload_mask_array(mask_result["binary_mask"])
            # except Exception as e:
            #     print(f"❌ Failed to upload mask array: {e}")
            #     mask_url = None

            # Prepare final results
            complete_results = {
                "meta": {
                    "classes": {
                        "0": "Mild",
                        "1": "Moderate",
                        "2": "Severe",
                        "3": "Very Severe"
                    },
                    "conf_threshold": segmentation_threshold
                },
                "predicts": [
                    {
                        "name": severity_label,
                        "confidence": severity_confidence,
                        "classes": severity_class,
                        # "segmentation": {
                        #     "acne_percentage": mask_result['acne_percentage'],
                        #     "mask_url": mask_url
                        # }
                    }
                ]
            },

            print(f"✅ Analysis completed!")
            return complete_results

        except Exception as e:
            print(f"❌ Error in complete analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "filename": uploaded_file.filename
            }


if __name__ == "__main__":
    # Example usage
    segmentation_model_path = "./weights/best_model.keras"
    severity_model_path = "./weights/best.pth"

    analyzer = CompleteAcneAnalyzer()
    print("🎉 Complete Acne Analysis Pipeline ready!")
"""Unit tests for the emotion detector; API replies are mocked."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import (
    EMOTION_NAMES,
    MODEL_ID,
    WATSON_EMOTION_URL,
    emotion_detector,
)


def make_api_response(scores, status_code=200):
    """Create a response-like mock using the Watson API response shape."""
    response = Mock()
    response.status_code = status_code
    response.json.return_value = {
        "emotionPredictions": [{"emotion": scores}],
    }
    return response


def scores_for(dominant):
    """Create sample scores with the requested dominant emotion."""
    scores = {emotion: 0.01 for emotion in EMOTION_NAMES}
    scores[dominant] = 0.96
    return scores


class TestEmotionDetection(unittest.TestCase):
    """Verify formatted results and error handling."""

    def check_dominant_emotion(self, sentence, dominant):
        response = make_api_response(scores_for(dominant))
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response,
        ):
            result = emotion_detector(sentence)
        self.assertEqual(result["dominant_emotion"], dominant)
        self.assertEqual(set(result), set(EMOTION_NAMES) | {"dominant_emotion"})
        self.assertTrue(all(isinstance(result[name], float) for name in EMOTION_NAMES))

    def test_joy(self):
        self.check_dominant_emotion("I am glad this happened", "joy")

    def test_anger(self):
        self.check_dominant_emotion("I am really mad about this", "anger")

    def test_disgust(self):
        self.check_dominant_emotion("I feel disgusted by this", "disgust")

    def test_sadness(self):
        self.check_dominant_emotion("I am so sad about this", "sadness")

    def test_fear(self):
        self.check_dominant_emotion(
            "I am really afraid that this will happen", "fear"
        )

    def test_request_parameters(self):
        response = make_api_response(scores_for("anger"))
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response,
        ) as mocked_post:
            emotion_detector("I am angry")

        args, kwargs = mocked_post.call_args
        self.assertEqual(args[0], WATSON_EMOTION_URL)
        self.assertEqual(kwargs["headers"]["grpc-metadata-mm-model-id"], MODEL_ID)
        self.assertEqual(
            kwargs["json"], {"raw_document": {"text": "I am angry"}}
        )

    def test_http_400_returns_none_values(self):
        response = make_api_response({}, status_code=400)
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response,
        ):
            result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])
        self.assertTrue(all(result[name] is None for name in EMOTION_NAMES))

    def test_unexpected_status_returns_none_values(self):
        response = make_api_response({}, status_code=503)
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response,
        ):
            result = emotion_detector("test")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()

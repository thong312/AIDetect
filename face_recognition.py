import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(image_pil):
    """Lấy embedding khuôn mặt bằng PyTorch."""
    try:
        face_tensor = mtcnn(image_pil)
        if face_tensor is None:
            return None
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_tensor).detach().cpu()
        return embedding
    except:
        return None

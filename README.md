# SEA-T2F
Multi-caption Text-to-Face Synthesis: Database and Algorithm

Requirements: python=3.7, pytorch=1.1.0, torchvision=0.3.0

1. Download data:
1) Download the CelebAText-HQ from https://drive.google.com/drive/folders/1IAb_iy6-soEGQWhbgu6cQODsIUJZpahC?usp=sharing, and extract them into data/CelebAText-HQ.
2) Download the Multi-modal CelebA-HQ from https://drive.google.com/drive/folders/1U1DvkFlcYJBUYpo8lmvlmywln5LEc0_O?usp=sharing, and extract them into data/Multi-modal.
3) Download the CelebAMask-HQ from https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv  and copy CelebAimg into data/CelebAText-HQ or data/Multi-modal CelebA-HQ.

2. Download pretrain model:
1)Download the pretriain model of CelebAText-HQ from https://drive.google.com/drive/folders/1XufcOo_I09h86ZR2M4UJ8WTVR7ARF87d, and extract them into /DAMSEencoders/CelebAText-HQ.
1)Download the pretriain model of Multi-modal CelebA-HQ from https://drive.google.com/drive/folders/1FN4q7xD1jKvXeG3Pd6wqieUlr52bVoIX, and extract them into /DAMSEencoders/CelebAText-HQ.

2. Download generative model:
1)Download the generative model of CelebAText-HQ from https://drive.google.com/drive/folders/1XufcOo_I09h86ZR2M4UJ8WTVR7ARF87d, and copy it into /models.
1)Download the generative model of Multi-modal CelebA-HQ from https://drive.google.com/drive/folders/1FN4q7xD1jKvXeG3Pd6wqieUlr52bVoIX, and copy it into /models.

3. Train:
python main.py --cfg cfg/train_face.yml --gpu 7

4. Evaluation:
 python main.py --cfg cfg/eval_face.yml --gpu 7

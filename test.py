import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

if __name__=="__main__":
    import sys
    from segment_anything import sam_model_registry, SamPredictor

    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    if False:
        # prepare image 1 and image 2
        image = cv2.imread('notebooks/images/truck.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = image  # truck.jpg from above
        image1_boxes = torch.tensor([
            [75, 275, 1725, 850],
            [425, 600, 700, 875],
            [1375, 550, 1650, 800],
            [75, 275, 1725, 850],
            [425, 600, 700, 875],
            [1375, 550, 1650, 800],
        ], device=sam.device)

        image1_point = torch.tensor([[500, 375], [1125, 625], [501, 376]],device=sam.device)
        image1_point_label = torch.tensor([1, 1, 1],device=sam.device)

        image1_point = image1_point[None,:,:].expand(image1_boxes.shape[0],-1,-1)
        image1_point_label = image1_point_label[None,:].expand(image1_boxes.shape[0],-1)
        image_masks=np.load("notebooks/cache/mask_input--img1.npy")[None,:,:]

        image2 = cv2.imread('notebooks/images/groceries.jpg')
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2_boxes = torch.tensor([
            [450, 170, 520, 350],
            [350, 190, 450, 350],
            [500, 170, 580, 350],
            [580, 170, 640, 350],
            [580, 170, 640, 350],
        ], device=sam.device)
        
        # create batched input
        from segment_anything.utils.transforms import ResizeLongestSide
        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

        def prepare_image(image, transform, device):
            image = transform.apply_image(image)
            image = torch.as_tensor(image, device=device.device) 
            return image.permute(2, 0, 1).contiguous()
        
        batched_input = [
            {
                'image': prepare_image(image1, resize_transform, sam), # (3, H0, W0)
                'boxes': resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]), # (3,4)
                'point_coords': resize_transform.apply_coords_torch(image1_point, image1.shape[:2]), # (3,2,2)
                'point_labels': image1_point_label,
        #          'mask_inputs': torch.tensor(image_masks),
                'original_size': image1.shape[:2]
            },
            {
                'image': prepare_image(image2, resize_transform, sam), # (3, H0, W0)
                'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]), # (5,4)
                'original_size': image2.shape[:2]
            }
        ]


        batched_output = sam(batched_input, multimask_output=False)

    else:
        image = cv2.imread('notebooks/images/truck.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_point = np.array([[500, 375]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )


# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology & Oriane Simeoni, valeo.ai
# ---------------------------------------------------------------------------------------------------

from typing import Optional

import cv2
import numpy as np
from mmseg_utils.datasets.builder import PIPELINES
from mmseg_utils.datasets.pipelines.formatting import ImageToTensor, to_tensor

# import Union, Tuple
from typing import Union, Tuple


@PIPELINES.register_module()
class ToRGB:
    def __call__(self, results):
        return self.transform(results)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to go from BGR to RGB.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        results['img'] = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
        return results


# ________________
# Modified version from mmcv to directly convert a tensor to float
# MAKE SURE YOU USE IT ONLY FOR IMAGES

@PIPELINES.register_module()
class ImageToTensorV2(ImageToTensor):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Required keys:

    - all these keys in `keys`

    Modified Keys:

    - all these keys in `keys`

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys: dict) -> None:
        super(ImageToTensorV2, self).__init__(keys)
        self.keys = keys

    def __call__(self, results):
        return self.transform(results)

    def transform(self, results: dict) -> dict:
        """Transform function to convert image in results to
        :obj:`torch.Tensor` and transpose the channel order.
        Args:
            results (dict): Result dict contains the image data to convert.
        Returns:
            dict: The result dict contains the image converted
            to :obj:``torch.Tensor`` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            results[key] = (to_tensor(img.copy()).permute(2, 0, 1)).contiguous() / 255.

        return results
    
@PIPELINES.register_module()
class DeterministCrop():
    """Random crop the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        # super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Generate a crop bounding box centered in the middle of the image.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """
            # Get the dimensions of the image
            img_h, img_w = img.shape[:2]

            # Calculate the start and end points of the crop box
            crop_h, crop_w = self.crop_size
            center_h, center_w = img_h // 2, img_w // 2

            crop_y1 = max(center_h - crop_h // 2, 0)
            crop_y2 = min(crop_y1 + crop_h, img_h)
            crop_x1 = max(center_w - crop_w // 2, 0)
            crop_x2 = min(crop_x1 + crop_w, img_w)

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img
    
    def __call__(self, results):
        return self.transform(results)

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

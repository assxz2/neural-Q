import os
from pathlib import Path

import cv2
import json
import tifffile
import zipfile
import numpy as np
from tqdm import tqdm

from skimage.filters import threshold_otsu


class Compressor(object):
    def __init__(self, tif_dir):
        """
        Compress original TIFF sequence in one directory into a compressed format
        
        1. Remove unrelated background float values
        2. Compresse sparse matrix with zip compress method
        3. Transfer TIFF sequences into readable video files
        """
        self._dir = tif_dir
        self._thresh = None
        self.photo_info = None
        self.max_value = None
        self.min_value = None
        self.gamma_value = None
        self.frames = tifffile.imread(tif_dir)
        self.width = self.frames[0].shape[1]
        self.height = self.frames[0].shape[0]
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32, 32))
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        # self.parse_photo_info()
    
    def gamma_correct(self, image, gamma):
        """
        Apply Gamma correction on input image
        
        :param image: numpy ndarray
        :type image: ndarray
        :param gamma: gamma value
        :type gamma: float
        """
        max_value = np.iinfo(image.dtype).max
        # Add small number for numeric stable
        gamma_correction = ((image / max_value) ** (1 / gamma)) * max_value + 0.5
        return gamma_correction.astype(image.dtype)
    
    def parse_photo_info(self):
        """
        There is a `display_and_comments.txt` automatic generated file in tiff directory,
        And I thought it will be informative for image converting.
        
            {'Channels': 
              [{'Name': 'Default',
                'DisplayMode': 1,
                'HistogramMax': 16383,
                'Max': 6994,
                'Gamma': 0.8805580899963568,
                'Min': 264,
                'Color': -1}],
             'Comments': {'Summary': ''}}
        """
        with open(os.path.join(self._dir, 'display_and_comments.txt'), 'r') as fp:
            photo_info = json.loads(fp.read())
            assert len(photo_info['Channels']) == 1, "Only handle single channel image now!"
            self.max_value = photo_info['Channels'][0]['Max']
            self.min_value = photo_info['Channels'][0]['Min']
            self.gamma_value = photo_info['Channels'][0]['Gamma']
            self.photo_info = photo_info
    
    def map_uint16_to_uint8(self, img, lower_bound=None, upper_bound=None):
        """
        Map a 16-bit image trough a lookup table to convert it to 8-bit.

        :param img: numpy.ndarray[np.uint16]
            image that should be mapped
        :param lower_bound: int, optional
            lower bound of the range that should be mapped to ``[0, 255]``,
            value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
            (defaults to ``numpy.min(img)``)
        :param upper_bound: int, optional
           upper bound of the range that should be mapped to ``[0, 255]``,
           value must be in the range ``[0, 65535]`` and larger than `lower_bound`
           (defaults to ``numpy.max(img)``)

        :rtype: numpy.ndarray[uint8]
        """
        if lower_bound is None:
            lower_bound = np.min(img)
        if upper_bound is None:
            upper_bound = np.max(img)
        if lower_bound >= upper_bound:
            raise ValueError(
                '"lower_bound" must be smaller than "upper_bound"')

        if not(0 <= lower_bound < 2**16) and lower_bound is not None:
            raise ValueError(
                '"lower_bound" must be in the range [0, 65535]')
        if not(0 <= upper_bound < 2**16) and upper_bound is not None:
            raise ValueError(
                '"upper_bound" must be in the range [0, 65535]')

        lut = np.concatenate([
            np.zeros(lower_bound, dtype=np.uint16),
            np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
            np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
        ])
        return lut[img].astype(np.uint8)
        
    def front_object(self, image):
        """
        Find out the tight ROI of foreground object
        
        :param image: numpy array of image
        :type image: ndarray
        """
        # Simple binarized image with OTSU methods
        thresh = threshold_otsu(image)
        binary = image > thresh
        binary_img = np.asarray(binary)
        dilated_img = cv2.dilate(binary_img, self.dilate_kernel)
        eroded_img = cv2.erode(dilated_img, self.erode_kernel)
        
        # find tight rectangle of front object shape
        contours, hierarchy = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        rotate_rect = cv2.minAreaRect(sorted_contours[0])
        rotate_box = cv2.boxPoints(rotate_rect)
        rotate_box = np.int0(rotate_box)
        x, y, w, h = cv2.boundingRect(rotate_box)
        delta = min(h, w) // 2
        ys = max(0, y-delta)
        xs = max(0, x-delta)
        ye = min(image.shape[0], y + h + delta)
        xe = min(image.shape[1], x + w + delta)
        roi_img = image[ys:ye, xs:xe]
        return roi_img
    
    def zero_background(self, image):
        """
        Zero out the background of image, return the masked original image
        
        :param image: the image to be masked
        :type image: ndarray
        """
        thresh = threshold_otsu(image)
        binary = image > thresh

        binary_img = np.asarray(binary, dtype=np.uint8)
        dilated_img = cv2.dilate(binary_img, self.dilate_kernel)
        eroded_img = cv2.erode(dilated_img, self.erode_kernel)

        contours, hierarchy = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        rotate_rect = cv2.minAreaRect(sorted_contours[0])

        rotate_box = cv2.boxPoints(rotate_rect)
        rotate_box = np.int0(rotate_box)
        x, y, w, h = cv2.boundingRect(rotate_box)
        delta = min(h, w) // 2
        ys = max(0, y-delta)
        xs = max(0, x-delta)
        ye = min(image.shape[0], y + h + delta)
        xe = min(image.shape[1], x + w + delta)
        roi_img = image[ys:ye, xs:xe]
        mask = np.zeros_like(image)
        mask[ys:ye, xs:xe] = 1
        return image * mask

    def create_video(self, out_path):
        """
        Create the video from read TIFF frame sequence
        
        :param out_path: the output file path for the video
        :type out_path: str
        """
        self.parse_photo_info()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            out_path,
            fourcc,
            25.0, (self.width, self.height), 0)
        
        print("Start writing TIFF sequence into single video file...")
        for frame in tqdm(self.frames):
            visual_image = self.gamma_correct(frame, self.gamma_value)
            uint8_image = self.map_uint16_to_uint8(visual_image, 
                                                   lower_bound=self.min_value,
                                                   upper_bound=self.max_value)
            video_writer.write(uint8_image)
        video_writer.release()
        print("Write the TIFF sequence {} into video {} successfully!".format(self._dir, out_path))

    def write_compressed_tiff(self, out_path, to8bit=False):
        """
        Write the converted tiff image sequence into single tiff with multi-frame
        
        :param out_path: the output file name inside zip file with absolute path
        :type param: str
        """
        with tifffile.TiffWriter(out_path) as tif:
            for frame in tqdm(self.frames):
                if to8bit:
                    visual_frame = self.gamma_correct(frame, self.gamma_value)
                    frame = self.map_uint16_to_uint8(visual_frame, lower_bound=self.min_value, upper_bound=self.max_value)
                tif.write(self.zero_background(frame), contiguous=True)
        zipfilename = out_path.replace('.tif', '.zip')
        with zipfile.ZipFile(zipfilename, 'w') as zf:
            zf.write(out_path, compress_type=zipfile.ZIP_DEFLATED)
        print("save compressed {} to {} successfully!".format(self._dir, out_path.replace('.tif', '.zip')))
        os.remove(out_path)
        print("Temporary file {} removed".format(out_path))

        
def uncompress_tiff_zip(zip_file, out_path):
    with zipfile.ZipFile(zip_file, 'r') as zf:
        if out_path is not None:
            zf.extractall(out_path)
        else:
            zf.extractall()
    print("Unzip compressed file to {}".format(out_path))


def compress_dataset(dataset_root, compressed_data_root):
    for p in tqdm(Path(dataset_root).rglob('**/*')):
        if p.is_dir() and len(os.listdir(p)) > 10 and '.tif' in os.listdir(p)[-1]:
            vp = str(p).replace(dataset_root, compressed_data_root)
            vp_parent = Path(vp).parent
            if not vp_parent.exists():
                vp_parent.mkdir(parents=True)
                print("{} created".format(str(vp_parent)))
            if not Path(vp+'.zip').exists():
                compressor = Compressor(str(p))
                try:
                    compressor.write_compressed_tiff(vp+'.tif')
                except Exception as e:
                    print("Abort {} video create! with Exception {}".format(vp, e))


if __name__ == '__main__':
    # compressor = Compressor("/data/drosophila/light-sheet/VNC/79029x39337/syx_2021-1-22_1-1")
    # compressor.write_compressed_tiff("./data/test.tif", to8bit=True)
    compress_dataset("/data", "/data/compressed_test_data")

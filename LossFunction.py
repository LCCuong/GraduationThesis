import numpy as np
from PIL import Image
from utils import IoU

class UntargetedLoss:
  def __init__(self, model, gt_label, gt_box, _lambda = 1e-3):
        self.model = model
        self.gt_label = gt_label
        self.gt_box = gt_box
        self._lambda = _lambda

  def __call__(self, image, values, prev_conf): # image is np.array with integer value from 0 - 255
        data = self.model(image)
        
        is_adver = False
        score_list = []
        adver_list = []
        f_score_list = []

        for idx, gt_box in enumerate(self.gt_box): # For each box in ground truth boxes
            candidates = []
            candidates_IoU = []
            for res in data:
                box = res[:4].astype(np.int64)
                cal_IoU = IoU(gt_box, box)
                if cal_IoU > 0.5:
                    candidates.append(res)
                    candidates_IoU.append(cal_IoU)

            if len(candidates) == 0:
                is_adver = is_adver or False
                score_list.append(0)
                adver_list.append(False)
                f_score_list.append(0)
                continue
            
            f_score = []
            classes = []

            for candidate in candidates:
                cls = int(candidate[5]) # Get class
                classes.append(cls)
                if np.absolute(cls - self.gt_label[idx]) < 1e-10:
                    f_score.append(candidate[4])
                else:
                    f_score.append(-candidate[4])

            classes = np.array(classes)
            adver = True # is_adver if only all the boxes with IOU > 0.5 have different classes from ground truth

            if np.any(np.absolute(classes - self.gt_label[idx]) < 1e-10):
                adver = False
            is_adver = is_adver or adver

            IOU_score = np.mean(np.log(1 - np.array(candidates_IoU) + 0.1))

            if adver:
                f_score_list.append(min(f_score))
            else:
                f_score_list.append(max(f_score))
            f_score = np.mean(f_score)

            L = IOU_score + 2 * f_score - prev_conf[idx] + np.linalg.norm(values[idx].flatten(), ord = 0) / (3 * len(values[idx]))
            score_list.append(L)
            adver_list.append(adver)
        # if is_adver:
        #   i = 0
        #   while os.path.isfile(f"res/perturbed{i}.jpg"):
        #     i += 1
        #   results[0].save(filename=f"res/perturbed{i}.jpg")
        
        L = np.sum(score_list)
        
        return is_adver, L, score_list, adver_list, f_score_list  
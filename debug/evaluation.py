from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import *


def main():
    predicted_depths = load_depth()
    test_files = get_img_paths()
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files)
    num_samples = len(im_files)
    ground_truths = []

    for t_id in range(num_samples):
        camera_id = cams[t_id]
        depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
        ground_truths.append(depth.astype(np.float32))

        depth = cv2.resize(predicted_depths[t_id],(im_sizes[t_id][1], im_sizes[t_id][0]),interpolation=cv2.INTER_LINEAR)
        predicted_depths[t_id] = depth

    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    rmse = np.zeros(num_samples, np.float32)
    rmse_log = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(len(ground_truths)):
        ground_depth = ground_truths[i]
        predicted_depth = predicted_depths[i]

        predicted_depth[predicted_depth < 1] = 1
        predicted_depth[predicted_depth > 50] = 50

        height, width = ground_depth.shape
        mask = np.logical_and(ground_depth > 1, ground_depth < 50)

        crop = np.array([0.40810811 * height, 0.99189189 * height,
                         0.03594771 * width, 0.96405229 * width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i] = compute_errors(ground_depth[mask],predicted_depth[mask])

        print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
              .format(i, abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i]))

    print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
    print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
           .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))


if __name__ == '__main__':
    main()
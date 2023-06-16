import ipdb
from tqdm import tqdm
import json
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
import shutil
import cv2
from skimage import io
from model import lambSH_layer, SfMNet
from utils.render_sphere_nm import render_sphere_nm
from utils.whdr import compute_whdr
from utils.diode_metrics import angular_error
import argparse

parser = argparse.ArgumentParser(description="InverseRenderNet++")
parser.add_argument(
    "--mode",
    type=str,
    default="video",
    choices=["singleframe","demo_im", "iiw", "diode","video"],
    help="testing mode",
)

parser.add_argument(
    "--model",
    type=str,
    default="model_ckpt",
    choices=["model_ckpt","diode_model_ckpt", "iiw_model_ckpt"],
    help="model option",
)

parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="fps",
)

#python video.py --input <path>/<file>.mp4 --fps <fps> --output <path>/

# test demo image
parser.add_argument("--image", type=str, default=None, help="Path to test image")
parser.add_argument("--mask", type=str, default=None, help="Path to image mask")

# input and output path
parser.add_argument("--input", type=str, required=True, help="Video File Input")
parser.add_argument("--output", type=str, required=True, help="Folder saving outputs")
args = parser.parse_args()


def rescale_2_zero_one(imgs):
    return imgs / 2.0 + 0.5


def srgb_to_rgb(srgb):
    """Taken from bell2014: sRGB -> RGB."""
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def irn_func(input_height, input_width):
    # define inputs

    inputs_var = tf.placeholder(tf.float32, (None, input_height, input_width, 3))
    masks_var = tf.placeholder(tf.float32, (None, input_height, input_width, 1))
    train_flag = tf.placeholder(tf.bool, ())
  
    albedos, shadow, nm_pred = SfMNet.SfMNet(
        inputs=inputs_var,
        is_training=train_flag,
        height=input_height,
        width=input_width,
        masks=masks_var,
        n_layers=30,
        n_pools=4,
        depth_base=32,
    )
  
    
    gamma = tf.constant(2.2)
    lightings, _ = SfMNet.comp_light(
        inputs_var, albedos, nm_pred, shadow, gamma, masks_var
    )

    # rescale
    albedos = rescale_2_zero_one(albedos) * masks_var
    shadow = rescale_2_zero_one(shadow)
    inputs = rescale_2_zero_one(inputs_var) * masks_var
 
    # visualise lighting on a sphere
    num_rendering = tf.shape(lightings)[0]
    nm_sphere = tf.constant(render_sphere_nm(100, 1), dtype=tf.float32)
    nm_sphere = tf.tile(nm_sphere, (num_rendering, 1, 1, 1))
    lighting_recon, _ = lambSH_layer.lambSH_layer(
        tf.ones_like(nm_sphere), nm_sphere, lightings, tf.ones_like(nm_sphere), 1.0
    )

    # recon shading map
    shading, _ = lambSH_layer.lambSH_layer(
        tf.ones_like(albedos), nm_pred, lightings, tf.ones_like(albedos), 1.0
    )

    return (
        albedos,
        shadow,
        nm_pred,
        lighting_recon,
        shading,
        inputs,
        inputs_var,
        masks_var,
        train_flag,
    )



def post_process_albedo_nm(
    albedos_val,
    nm_pred_val,
    ori_width,
    ori_height,
    resize=False,
):
    # post-process results
    results = {}

    if resize:
        results.update(
            dict(albedos=cv2.resize(albedos_val[0], (ori_width, ori_height)))
        )

        results.update(
            dict(nm_pred=cv2.resize(nm_pred_val[0], (ori_width, ori_height)))
        )
    else:
        results.update(dict(albedos=albedos_val[0]))

        results.update(dict(nm_pred=nm_pred_val[0]))

    return results

def post_process(
    albedos_val,
    shading_val,
    shadow_val,
    lighting_recon_val,
    nm_pred_val,
    ori_width,
    ori_height,
    resize=True,
):
    # post-process results
    results = {}

    if resize:
        results.update(
            dict(albedos=cv2.resize(albedos_val[0], (ori_width, ori_height)))
        )

        results.update(
            dict(shading=cv2.resize(shading_val[0], (ori_width, ori_height)))
        )

        results.update(
            dict(shadow=cv2.resize(shadow_val[0, :, :, 0], (ori_width, ori_height)))
        )

        results.update(dict(lighting_recon=lighting_recon_val[0]))

        results.update(
            dict(nm_pred=cv2.resize(nm_pred_val[0], (ori_width, ori_height)))
        )
    else:
        results.update(dict(albedos=albedos_val[0]))

        results.update(dict(shading=shading_val[0]))

        results.update(dict(shadow=shadow_val[0, ..., 0]))

        results.update(dict(lighting_recon=lighting_recon_val[0]))

        results.update(dict(nm_pred=nm_pred_val[0]))

    return results


def saving_result(results, dst_dir, prefix=""):
    img = np.uint8(results["img"])
    albedos = np.uint8(results["albedos"] * 255.0)
    #shading = np.uint8(results["shading"] * 255.0)
    #shadow = np.uint8(results["shadow"] * 255.0)
    #lighting_recon = np.uint8(results["lighting_recon"] * 255.0)
    nm_pred = np.uint8(results["nm_pred"] * 255.0)

    # save images
    input_path = os.path.join(dst_dir, prefix + "img.png")
    io.imsave(input_path, img)
    nm_pred_path = os.path.join(dst_dir, prefix + "nm_pred.png")
    io.imsave(nm_pred_path, nm_pred)
    albedo_path = os.path.join(dst_dir, prefix + "albedo.png")
    io.imsave(albedo_path, albedos)
    #shading_path = os.path.join(dst_dir, prefix + "shading.png")
    #io.imsave(shading_path, shading)
    #shadow_path = os.path.join(dst_dir, prefix + "shadow.png")
    #io.imsave(shadow_path, shadow)
    #lighting_path = os.path.join(dst_dir, prefix + "lighting.png")
    #io.imsave(lighting_path, lighting_recon)
    pass


def rescale_img(img):
    img_h, img_w = img.shape[:2]
    if img_h > img_w:
        scale = img_w / 200
        new_img_h = np.int32(img_h / scale)
        new_img_w = 200

        img = cv2.resize(img, (new_img_w, new_img_h))
    else:
        scale = img_h / 200
        new_img_w = np.int32(img_w / scale)
        new_img_h = 200

        img = cv2.resize(img, (new_img_w, new_img_h))

    return img, (img_h, img_w), (new_img_h, new_img_w)

def write_video(filename, output_list, fps):
    assert (len(output_list) > 0)
    h, w = output_list[0].shape[0], output_list[0].shape[1]
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for img in output_list:
        writer.write(img)
    writer.release()

    return



if args.mode == "demo_im":
    assert args.image is not None and args.mask is not None

    # read in images
    img_path = args.image
    mask_path = args.mask

    img = io.imread(img_path)
    mask = io.imread(mask_path)

    input_height = 200
    input_width = 200

    ori_img, (ori_height, ori_width), (input_height, input_width) = rescale_img(img)

    # run inverse rendering
    (
        albedos,
        shadow,
        nm_pred,
        lighting_recon,
        shading,
        inputs,
        inputs_var,
        masks_var,
        train_flag,
    ) = irn_func(input_height, input_width)

    # load model and run session
    model_path = tf.train.get_checkpoint_state(args.model).model_checkpoint_path
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # evaluation
    dst_dir = args.output
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir)

    imgs = np.float32(ori_img) / 255.0
    imgs = srgb_to_rgb(imgs)
    imgs = imgs * 2.0 - 1.0
    imgs = imgs[None]
    mask = cv2.resize(mask, (input_width, input_height), cv2.INTER_NEAREST)
    img_masks = np.ones((1, input_height, input_width, 1), np.bool)# np.float32(mask == 255)[None, ..., None]
    imgs *= img_masks
    [
        albedos_val,
        nm_pred_val,
        shadow_val,
        lighting_recon_val,
        shading_val,
        inputs_val,
    ] = sess.run(
        [albedos, nm_pred, shadow, lighting_recon, shading, inputs],
        feed_dict={inputs_var: imgs, masks_var: img_masks, train_flag: False},
    )

    # post-process results
    results = post_process(
        albedos_val,
        shading_val,
        shadow_val,
        lighting_recon_val,
        nm_pred_val,
        ori_width,
        ori_height,
    )

    # rescale albedo and normal
    results["albedos"] = (results["albedos"] - results["albedos"].min()) / (
        results["albedos"].max() - results["albedos"].min()
    )

    results["nm_pred"] = (results["nm_pred"] + 1.0) / 2.0
    results["img"] = img

    saving_result(results, dst_dir)


elif args.mode == "singleframe":
    assert args.image is not None

    # read in images
    img_path = args.image
    #mask_path = args.mask

    img = io.imread(img_path)
    #mask = io.imread(mask_path)

    h, w, c = img.shape

    input_height = h
    input_width = w

    ori_img, (ori_height, ori_width), (input_height, input_width) = rescale_img(img)

    # run inverse rendering
    (
        albedos,
        shadow,
        nm_pred,
        lighting_recon,
        shading,
        inputs,
        inputs_var,
        masks_var,
        train_flag,
    ) = irn_func(input_height, input_width)

    # load model and run session
    model_path = tf.train.get_checkpoint_state(args.model).model_checkpoint_path
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # evaluation
    dst_dir = args.output
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir)

    imgs = np.float32(ori_img) / 255.0
    imgs = srgb_to_rgb(imgs)
    imgs = imgs * 2.0 - 1.0
    imgs = imgs[None]
    #mask = cv2.resize(mask, (input_width, input_height), cv2.INTER_NEAREST)
    img_masks = np.ones((1, input_height, input_width, 1), np.bool)
    imgs *= img_masks
    [
        albedos_val,
        nm_pred_val,
        shadow_val,
        lighting_recon_val,
        shading_val,
        inputs_val,
    ] = sess.run(
        [albedos, nm_pred, shadow, lighting_recon, shading, inputs],
        feed_dict={inputs_var: imgs, masks_var: img_masks, train_flag: False},
    )

    # post-process results
    results = post_process(
        albedos_val,
        shading_val,
        shadow_val,
        lighting_recon_val,
        nm_pred_val,
        ori_width,
        ori_height,
    )

    # rescale albedo and normal
    results["albedos"] = (results["albedos"] - results["albedos"].min()) / (
        results["albedos"].max() - results["albedos"].min()
    )

    results["nm_pred"] = (results["nm_pred"] + 1.0) / 2.0
    results["img"] = img

    saving_result(results, dst_dir)
    
elif args.mode == "video":
    assert args.input is not None and args.output is not None 

    fps = args.fps
    videofile = args.input
    # get input
    #path_lists, scene_names = load_video_paths(args)

    # prepare output folder
    #os.makedirs(args.output, exist_ok=True)
    
    model_path = tf.train.get_checkpoint_state(args.model).model_checkpoint_path    
    # evaluation
    dst_dir = args.output
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir)
        
    cap = cv2.VideoCapture(videofile)

    output_albedo_list = []
    output_nm_list = []
    i = 1
    while (cap.isOpened()):
        tf.reset_default_graph()
        # Capture frame-by-frame
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
     
     
        if frame is not None:     
            # Display the resulting frame
            #cv2.imshow('Frame', frame)

            # START PROCESS
            # read in images

            img = frame
            #mask = io.imread(mask_path)

            h, w, c = img.shape

            input_height = h
            input_width = w


            #print(" input_height : " , input_height)
            #print(" input_width : " , input_width)

            ori_img, (ori_height, ori_width), (input_height, input_width) = rescale_img(img)

            # run inverse rendering
            (
                albedos,
                shadow,
                nm_pred,
                lighting_recon,
                shading,
                inputs,
                inputs_var,
                masks_var,
                train_flag,
            ) = irn_func(input_height, input_width)

            # load model and run session

            sess = tf.InteractiveSession()
            saver = tf.train.Saver()
            saver.restore(sess, model_path)



            imgs = np.float32(ori_img) / 255.0
            imgs = srgb_to_rgb(imgs)
            imgs = imgs * 2.0 - 1.0
            imgs = imgs[None]
            #mask = cv2.resize(mask, (input_width, input_height), cv2.INTER_NEAREST)
            img_masks = np.ones((1, input_height, input_width, 1), np.bool)
            imgs *= img_masks
            [
                albedos_val,
                nm_pred_val,
                shadow_val,
                lighting_recon_val,
                shading_val,
                inputs_val,
            ] = sess.run(
                [albedos, nm_pred, shadow, lighting_recon, shading, inputs],
                feed_dict={inputs_var: imgs, masks_var: img_masks, train_flag: False},
            )

            # post-process results
            results = post_process_albedo_nm(
                albedos_val,
                nm_pred_val,
                ori_width,
                ori_height,
            )

            # rescale albedo and normal
            results["albedos"] = (results["albedos"] - results["albedos"].min()) / (
                results["albedos"].max() - results["albedos"].min()
            )

            results["nm_pred"] = (results["nm_pred"] + 1.0) / 2.0
            results["img"] = img

            # save output
            #output_albedo_name = os.path.join(args.output, scene_names[i] + '.mp4')
            #output_nm_name = os.path.join(args.output, scene_names[i] + '.mp4')

            albedos = np.uint8(results["albedos"] * 255.0)
            nm_pred = np.uint8(results["nm_pred"] * 255.0)
            
            output_albedo_list.append(albedos)
            output_nm_list.append(nm_pred)
            print(" Processing : " , i)
            i +=1

        
        
        # END PROCESS
        
        # define q as the exit button
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
     
    # release the video capture object
    cap.release()
    output_albedo_name = os.path.join(args.output, 'albedo.mp4')
    output_nm_name = os.path.join(args.output, 'normals.mp4')
    
    print(" Writing: " + output_albedo_name)
    write_video(output_albedo_name,output_albedo_list,fps)
    
    print(" Writing: " + output_nm_name)
    write_video(output_nm_name,output_nm_list,fps)    
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()    
    print(" DONE: ")
    sess.close()

    
    
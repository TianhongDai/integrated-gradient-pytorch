import numpy as np
import cv2

G = [0, 255, 0]
R = [0, 0, 255]

def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)

def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2, plot_distribution=False):
    m = compute_threshold_by_top_percentage(attributions, percentage=100-clip_above_percentile, plot_distribution=plot_distribution)
    e = compute_threshold_by_top_percentage(attributions, percentage=100-clip_below_percentile, plot_distribution=plot_distribution)
    if(m == e):
        transformed = attributions
    else:
        transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
        transformed *= np.sign(attributions)
        transformed *= (transformed >= low)
        transformed = np.clip(transformed, 0.0, 1.0)
    
    return transformed

def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    if(attribution_sum == 0):
        threshold = 0
    else:
        sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
 
        cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
        threshold_idx = np.where(cum_sum >= percentage)[0][0]
        threshold = sorted_attributions[threshold_idx]
        if plot_distribution:
            raise NotImplementedError 
    
    return threshold

def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError

def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)

def visualize(attributions, image, positive_channel=G, negative_channel=R, polarity='positive', \
                clip_above_percentile=99.9, clip_below_percentile=0, morphological_cleanup=False, \
                structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=90, overlay=True, \
                mask_mode=False, plot_distribution=False):
    if polarity == 'both':
        #Primero los positivos
        attributions1 = polarity_function(attributions, polarity='positive')
        channel1 = G
    
        # convert the attributions to the gray scale
        attributions1 = convert_to_gray_scale(attributions1)
        attributions1 = linear_transform(attributions1, clip_above_percentile, clip_below_percentile, 0.0, plot_distribution=plot_distribution)
        attributions_mask1 = attributions1.copy()
        if morphological_cleanup:
            raise NotImplementedError
        if outlines:
            raise NotImplementedError
        attributions1 = np.expand_dims(attributions1, 2) * channel1
        if overlay:
            if mask_mode == False:
                attributions1 = overlay_function(attributions1, image)
            else:
                attributions1 = np.expand_dims(attributions_mask1, 2)
                attributions1 = np.clip(attributions1 * image, 0, 255)
                attributions1 = attributions1[:, :, (2, 1, 0)]

        #Segundo los negativos
        attributions2 = polarity_function(attributions, polarity='negative')
        attributions2 = np.abs(attributions2)
        channel2 = R
    
        # convert the attributions to the gray scale
        attributions2 = convert_to_gray_scale(attributions2)
        attributions2 = linear_transform(attributions2, clip_above_percentile, clip_below_percentile, 0.0, plot_distribution=plot_distribution)
        attributions_mask2 = attributions2.copy()
        if morphological_cleanup:
            raise NotImplementedError
        if outlines:
            raise NotImplementedError
        attributions2 = np.expand_dims(attributions2, 2) * channel2
        if overlay:
            if mask_mode == False:
                attributions2 = overlay_function(attributions2, image)
            else:
                attributions2 = np.expand_dims(attributions_mask2, 2)
                attributions2 = np.clip(attributions2 * image, 0, 255)
                attributions2 = attributions2[:, :, (2, 1, 0)]
        
        attributionsr = attributions1 + attributions2


    elif polarity == 'positive':
        attributions1 = polarity_function(attributions, polarity='positive')
        channel1 = G
    
        # convert the attributions to the gray scale
        attributions1 = convert_to_gray_scale(attributions1)
        attributions1 = linear_transform(attributions1, clip_above_percentile, clip_below_percentile, 0.0, plot_distribution=plot_distribution)
        attributions_mask1 = attributions1.copy()
        if morphological_cleanup:
            raise NotImplementedError
        if outlines:
            raise NotImplementedError
        attributions1 = np.expand_dims(attributions1, 2) * channel1
        if overlay:
            if mask_mode == False:
                attributions1 = overlay_function(attributions1, image)
            else:
                attributions1 = np.expand_dims(attributions_mask1, 2)
                attributions1 = np.clip(attributions1 * image, 0, 255)
                attributions1 = attributions1[:, :, (2, 1, 0)]
        attributionsr = attributions1
    else :
        raise NotImplementedError


    return attributionsr

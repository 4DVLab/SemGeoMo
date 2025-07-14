import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from ....data_loaders import humanml_utils


MAX_LINE_LENGTH = 20


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(8, 8), fps=120, radius=4,
                   vis_mode='default', gt_frames=[], handshake_size=0, blend_size=0, step_sizes=[], lengths = [], joints2=None, painting_features=[], guidance=None,
                   pointcloud=None,person2=None, index_list=None):
    matplotlib.use('Agg')
    """
    A wrapper around explicit_plot_3d_motion that 
    uses gt_frames to determine the colors of the frames
    """
    data = joints.copy().reshape(len(joints), -1, 3)
    frames_number = data.shape[0]
    frame_colors = ['blue' if index in gt_frames else 'orange' for index in range(frames_number)]
    if vis_mode == 'unfold':
        frame_colors = ['purple'] *handshake_size + ['blue']*blend_size + ['orange'] *(120-handshake_size*2-blend_size*2) +['orange']*blend_size
        frame_colors = ['orange'] *(120-handshake_size-blend_size) + ['orange']*blend_size + frame_colors*1024
    elif vis_mode == 'unfold_arb_len':
        for ii, step_size in enumerate(step_sizes):
            if ii == 0:
                frame_colors = ['orange']*(step_size - handshake_size - blend_size) + ['orange']*blend_size + ['purple'] * (handshake_size//2)
                continue
            if ii == len(step_sizes)-1:
                frame_colors += ['purple'] * (handshake_size//2) + ['orange'] * blend_size + ['orange'] * (lengths[ii] - handshake_size - blend_size)
                continue
            frame_colors += ['purple'] * (handshake_size // 2) + ['orange'] * blend_size + ['orange'] * (
                            lengths[ii] - 2 * handshake_size - 2 * blend_size) + ['orange'] * blend_size + \
                            ['purple'] * (handshake_size // 2)
    elif vis_mode == 'gt':
        frame_colors = ['blue'] * frames_number
    explicit_plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=figsize, fps=fps, radius=radius, 
                            vis_mode=vis_mode, frame_colors=frame_colors, joints2=joints2, painting_features=painting_features, guidance=guidance,
                            pc=pointcloud,person2=person2,index_list=index_list)


def plot_3d_motion_z(save_path, kinematic_tree, joints, title, dataset, figsize=(8, 8), fps=120, radius=4,
                   vis_mode='default', gt_frames=[], handshake_size=0, blend_size=0, step_sizes=[], lengths = [], joints2=None, painting_features=[], guidance=None,
                   pointcloud=None,person2=None, index_list=None):
    matplotlib.use('Agg')
    """
    A wrapper around explicit_plot_3d_motion that 
    uses gt_frames to determine the colors of the frames
    """
    data = joints.copy().reshape(len(joints), -1, 3)
    frames_number = data.shape[0]
    frame_colors = ['blue' if index in gt_frames else 'orange' for index in range(frames_number)]
    if vis_mode == 'unfold':
        frame_colors = ['purple'] *handshake_size + ['blue']*blend_size + ['orange'] *(120-handshake_size*2-blend_size*2) +['orange']*blend_size
        frame_colors = ['orange'] *(120-handshake_size-blend_size) + ['orange']*blend_size + frame_colors*1024
    elif vis_mode == 'unfold_arb_len':
        for ii, step_size in enumerate(step_sizes):
            if ii == 0:
                frame_colors = ['orange']*(step_size - handshake_size - blend_size) + ['orange']*blend_size + ['purple'] * (handshake_size//2)
                continue
            if ii == len(step_sizes)-1:
                frame_colors += ['purple'] * (handshake_size//2) + ['orange'] * blend_size + ['orange'] * (lengths[ii] - handshake_size - blend_size)
                continue
            frame_colors += ['purple'] * (handshake_size // 2) + ['orange'] * blend_size + ['orange'] * (
                            lengths[ii] - 2 * handshake_size - 2 * blend_size) + ['orange'] * blend_size + \
                            ['purple'] * (handshake_size // 2)
    elif vis_mode == 'gt':
        frame_colors = ['blue'] * frames_number
    explicit_plot_3d_motion_z(save_path, kinematic_tree, joints, title, dataset, figsize=figsize, fps=fps, radius=radius, 
                            vis_mode=vis_mode, frame_colors=frame_colors, joints2=joints2, painting_features=painting_features, guidance=guidance,
                            pc=pointcloud,person2=person2,index_list=index_list)
                            
                            
def explicit_plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(8, 8), fps=120, radius=4, 
                            vis_mode="default", frame_colors=[], joints2=None, painting_features=[], guidance=None,
                            pc=None,person2=None,index_list=None):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    index_list = index_list

    if type(title) == str:
        title = ["\n".join(wrap(title, 20))]
    elif type(title) == list:
        title = ["\n".join(wrap(t, 20)) for t in title]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title[0], fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    data2 = None
    if joints2 is not None:
        data2 = joints2.copy().reshape(len(joints), -1, 3)
    if guidance is not None:
        guidance['joint'] = guidance['joint'].copy().reshape(len(guidance['joint']), -1, 3)
        guidance['mask'] = guidance['mask'].copy().reshape(len(guidance['mask']), -1, 3)
    if person2 is not None:
        person2 = person2.reshape(len(joints), -1, 3)
        # print(person2.shape)

    # preparation related to specific datasets
    if dataset == "kit":
        data *= 0.003  # scale for visualization
    elif dataset in [ 'humanml',"imhoi"]:
        data *= 1.3  # scale for visualization
        if data2 is not None:
            data2 *= 1.3
        if guidance is not None:
            guidance['joint'] *= 1.3
    elif dataset in ["humanact12", "uestc"]:
        data *= -1.5  # reverse axes, scale for visualization
    elif dataset in ['humanact12', 'uestc', 'amass']:
        data *= -1.5 # reverse axes, scale for visualization
    elif dataset =='babel':
        data *= 1.3
    


    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    if data2 is not None and person2 is None:
        MINS = np.concatenate((data, data2)).min(axis=0).min(axis=0)
        MAXS = np.concatenate((data, data2)).max(axis=0).max(axis=0)
    elif data2 is not None and person2 is not None:
        MINS = np.concatenate((data, data2,person2)).min(axis=0).min(axis=0)
        MAXS = np.concatenate((data, data2,person2)).max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]
    colors_green = ["#008000", "#00FF00", "#32CD32", "#228B22", "#ADFF2F"] #for interactive_person_2
    colors_upper_body = colors_blue[:2] + colors_orange[2:]

    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "green":colors_green,"upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0].copy()

    if pc is not None:
        m = min(pc.shape[0],frame_number)
        #print(m)
        pointcloud = np.array(pc).copy()
        pointcloud = pointcloud[:m,...]
        pointcloud[:, :, 1] -= height_offset
        pointcloud[..., 0] -= data[:, 0:1, 0]
        pointcloud[..., 2] -= data[:, 0:1, 2]


    # Reduce data2 first before overriding root position with zeros
    if data2 is not None:
        data2[:, :, 1] -= height_offset
        data2[..., 0] -= data[:, 0:1, 0]
        data2[..., 2] -= data[:, 0:1, 2]


    if guidance is not None:
        guidance['joint'][:, :, 1] -= height_offset
        guidance['joint'][..., 0] -= data[:, 0:1, 0]
        guidance['joint'][..., 2] -= data[:, 0:1, 2]

    if person2 is not None:
        person2[:, :, 1] -= height_offset
        person2[..., 0] -= data[:, 0:1, 0]
        person2[..., 2] -= data[:, 0:1, 2]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
        
    def update(index):
        #ax.lines.clear()
        #ax.collections.clear()
        
        ax.cla()  # Clear the axes, instead of ax.lines.clear()
        init()

        
        ax.view_init(elev=20, azim=80, vertical_axis = 'y')###
        ax.dist = 7.5
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 2], MAXS[2] - trajec[index, 2])
        used_colors = colors_dict[frame_colors[index]] if (index < len(frame_colors)) else colors_dict["blue"]
        other_colors = colors_purple
        color2s = colors_green
        for i, (chain, color, other_color,color_2) in enumerate(zip(kinematic_tree, used_colors, other_colors,color2s)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
            num = index // 10
            #index_number = index_list["target_contact"][num]
            #if index_number in chain:
                #ax.scatter(data[index, index_number, 0], data[index, index_number, 1], data[index, index_number, 2], color='black', s=30)
            # # index_number = index_list["target_contact_2"][num]
            # # if index_number in chain:
            # #     ax.scatter(data[index, index_number, 0], data[index, index_number, 1], data[index, index_number, 2], color='blue', s=30)
            # index_number = index_list["target_far"][num]
            # if index_number in chain:
            #     ax.scatter(data[index, index_number, 0], data[index, index_number, 1], data[index, index_number, 2], color='blue', s=30)
            
            if data2 is not None:
                ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=other_color)
            
            if person2 is not None:
                ax.plot3D(person2[index, chain, 0], person2[index, chain, 1], person2[index, chain, 2], linewidth=linewidth, color=color_2)
                # index_number = index_list["interact_contact"][num]
                # if index_number in chain:
                #     ax.scatter(person2[index, index_number, 0], person2[index, index_number, 1], person2[index, index_number, 2], color='black', s=30)
                # index_number = index_list["interact_far"][num]
                # if index_number in chain:
                #     ax.scatter(person2[index, index_number, 0], person2[index, index_number, 1], person2[index, index_number, 2], color='blue', s=30)
        
        def plot_root_horizontal():
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 1]), trajec[:index, 2] - trajec[index, 2], linewidth=2.0,
                      color=used_colors[0])
        
        def plot_root():
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], trajec[:index, 1], trajec[:index, 2] - trajec[index, 2], linewidth=2.0,
                      color=used_colors[0])
        
        def plot_feature(feature):
            # trajectory = Line3DCollection(joints[:,0])
            if feature in humanml_utils.HML_JOINT_NAMES:
                feat_index = humanml_utils.HML_JOINT_NAMES.index(feature)
                ax.plot3D(data[:index+1, feat_index, 0] + (trajec[:index+1, 0] - trajec[index, 0]),
                          data[:index+1, feat_index, 1],
                          data[:index+1, feat_index, 2] + (trajec[:index+1, 2] - trajec[index, 2]), linewidth=2.0,
                        color=used_colors[0])
                
        def plot_global_joint(guidance_motion, mask):
            #
            #assert guidance_motion.shape == mask.shape
            num_joint = guidance_motion.shape[1]
            guidance = guidance_motion[index]
            mask = mask[index]
             
            #future_guidance[..., 0] = future_guidance[..., 0] #+ (np.tile(np.reshape(trajec[:, 0], (-1,1)), (1, num_joint)) - np.reshape(trajec[index, 0], (1,-1)))
            #future_guidance[..., 2] = future_guidance[..., 2] #+ (np.tile(np.reshape(trajec[:, 2], (-1,1)), (1, num_joint)) - np.reshape(trajec[index, 2], (1,-1)))
            guidance = guidance[mask.astype(bool)].reshape(-1, 3)
            ax.scatter(guidance[:, 0], guidance[:, 1], guidance[:, 2], color="#00FFFF")

        if 'root_horizontal' in painting_features:
            plot_root_horizontal()
        if 'root' in painting_features:
            plot_root()
        if 'global_joint' in painting_features:
            #assert guidance is not None
            #plot_global_joint(guidance['joint'], guidance['mask'])
            pass

        for feat in painting_features:
            plot_feature(feat)
        
        if pointcloud is not None:
            cloud = pointcloud[index]
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color="gray", s=1, alpha=0.5)
            num = index // 10
            # index = index_list["interact_contact"][num]
            # contact = cloud[index]
            # ax.scatter(contact[:, 0], contact[:, 1],contact[:, 2], color='red', s=10, alpha=0.5)
            # index = index_list["interact_far"][num]
            # far = cloud[index]
            # ax.scatter(far[:, 0], far[:, 1],far[:, 2], color='blue', s=10, alpha=0.5)

            

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 // fps, repeat=False)

    ani.save(save_path, fps=fps)

    plt.close()


def explicit_plot_3d_motion_z(save_path, kinematic_tree, joints, title, dataset, figsize=(8, 8), fps=120, radius=4, 
                            vis_mode="default", frame_colors=[], joints2=None, painting_features=[], guidance=None,
                            pc=None,person2=None,index_list=None):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    index_list = index_list

    if type(title) == str:
        title = ["\n".join(wrap(title, 20))]
    elif type(title) == list:
        title = ["\n".join(wrap(t, 20)) for t in title]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title[0], fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, maxy, minz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, maxy, minz], [maxx, maxy, minz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    data2 = None
    if joints2 is not None:
        data2 = joints2.copy().reshape(len(joints), -1, 3)
    if guidance is not None:
        guidance['joint'] = guidance['joint'].copy().reshape(len(guidance['joint']), -1, 3)
        guidance['mask'] = guidance['mask'].copy().reshape(len(guidance['mask']), -1, 3)
    if person2 is not None:
        person2 = person2.reshape(len(joints), -1, 3)
        # print(person2.shape)

    # preparation related to specific datasets
    if dataset == "kit":
        data *= 0.003  # scale for visualization
    elif dataset in [ 'humanml',"imhoi"]:
        data *= 1.3  # scale for visualization
        if data2 is not None:
            data2 *= 1.3
        if guidance is not None:
            guidance['joint'] *= 1.3
    elif dataset in ["humanact12", "uestc"]:
        data *= -1.5  # reverse axes, scale for visualization
    elif dataset in ['humanact12', 'uestc', 'amass']:
        data *= -1.5 # reverse axes, scale for visualization
    elif dataset =='babel':
        data *= 1.3
    


    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    if data2 is not None and person2 is None:
        MINS = np.concatenate((data, data2)).min(axis=0).min(axis=0)
        MAXS = np.concatenate((data, data2)).max(axis=0).max(axis=0)
    elif data2 is not None and person2 is not None:
        MINS = np.concatenate((data, data2,person2)).min(axis=0).min(axis=0)
        MAXS = np.concatenate((data, data2,person2)).max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]
    colors_green = ["#008000", "#00FF00", "#32CD32", "#228B22", "#ADFF2F"] #for interactive_person_2
    colors_upper_body = colors_blue[:2] + colors_orange[2:]

    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "green":colors_green,"upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0].copy()

    if pc is not None:
        m = min(pc.shape[0],frame_number)
        #print(m)
        pointcloud = np.array(pc).copy()
        pointcloud = pointcloud[:m,...]
        pointcloud[:, :, 1] -= height_offset
        pointcloud[..., 0] -= data[:, 0:1, 0]
        pointcloud[..., 2] -= data[:, 0:1, 2]


    # Reduce data2 first before overriding root position with zeros
    if data2 is not None:
        data2[:, :, 1] -= height_offset
        data2[..., 0] -= data[:, 0:1, 0]
        data2[..., 2] -= data[:, 0:1, 2]


    if guidance is not None:
        guidance['joint'][:, :, 1] -= height_offset
        guidance['joint'][..., 0] -= data[:, 0:1, 0]
        guidance['joint'][..., 2] -= data[:, 0:1, 2]

    if person2 is not None:
        person2[:, :, 1] -= height_offset
        person2[..., 0] -= data[:, 0:1, 0]
        person2[..., 2] -= data[:, 0:1, 2]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
        
    def update(index):
        ax.lines.clear()
        ax.collections.clear()
        
        ax.view_init(elev=20, azim=80, vertical_axis = 'z')###
        ax.dist = 7.5
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], MINS[1] - trajec[index, 1], MAXS[1] - trajec[index, 1],0)
        used_colors = colors_dict[frame_colors[index]] if (index < len(frame_colors)) else colors_dict["blue"]
        other_colors = colors_purple
        color2s = colors_green
        for i, (chain, color, other_color,color_2) in enumerate(zip(kinematic_tree, used_colors, other_colors,color2s)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
            num = index // 10
            #index_number = index_list["target_contact"][num]
            #if index_number in chain:
                #ax.scatter(data[index, index_number, 0], data[index, index_number, 1], data[index, index_number, 2], color='black', s=30)
            # # index_number = index_list["target_contact_2"][num]
            # # if index_number in chain:
            # #     ax.scatter(data[index, index_number, 0], data[index, index_number, 1], data[index, index_number, 2], color='blue', s=30)
            # index_number = index_list["target_far"][num]
            # if index_number in chain:
            #     ax.scatter(data[index, index_number, 0], data[index, index_number, 1], data[index, index_number, 2], color='blue', s=30)
            
            if data2 is not None:
                ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=other_color)
            
            if person2 is not None:
                ax.plot3D(person2[index, chain, 0], person2[index, chain, 1], person2[index, chain, 2], linewidth=linewidth, color=color_2)
                # index_number = index_list["interact_contact"][num]
                # if index_number in chain:
                #     ax.scatter(person2[index, index_number, 0], person2[index, index_number, 1], person2[index, index_number, 2], color='black', s=30)
                # index_number = index_list["interact_far"][num]
                # if index_number in chain:
                #     ax.scatter(person2[index, index_number, 0], person2[index, index_number, 1], person2[index, index_number, 2], color='blue', s=30)
        
        def plot_root_horizontal():
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 1]), trajec[:index, 2] - trajec[index, 2], linewidth=2.0,
                      color=used_colors[0])
        
        def plot_root():
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], trajec[:index, 1], trajec[:index, 2] - trajec[index, 2], linewidth=2.0,
                      color=used_colors[0])
        
        def plot_feature(feature):
            # trajectory = Line3DCollection(joints[:,0])
            if feature in humanml_utils.HML_JOINT_NAMES:
                feat_index = humanml_utils.HML_JOINT_NAMES.index(feature)
                ax.plot3D(data[:index+1, feat_index, 0] + (trajec[:index+1, 0] - trajec[index, 0]),
                          data[:index+1, feat_index, 1],
                          data[:index+1, feat_index, 2] + (trajec[:index+1, 2] - trajec[index, 2]), linewidth=2.0,
                        color=used_colors[0])
                
        def plot_global_joint(guidance_motion, mask):
            #
            #assert guidance_motion.shape == mask.shape
            num_joint = guidance_motion.shape[1]
            guidance = guidance_motion[index]
            mask = mask[index]
             
            #future_guidance[..., 0] = future_guidance[..., 0] #+ (np.tile(np.reshape(trajec[:, 0], (-1,1)), (1, num_joint)) - np.reshape(trajec[index, 0], (1,-1)))
            #future_guidance[..., 2] = future_guidance[..., 2] #+ (np.tile(np.reshape(trajec[:, 2], (-1,1)), (1, num_joint)) - np.reshape(trajec[index, 2], (1,-1)))
            guidance = guidance[mask.astype(bool)].reshape(-1, 3)
            ax.scatter(guidance[:, 0], guidance[:, 1], guidance[:, 2], color="#00FFFF")

        if 'root_horizontal' in painting_features:
            plot_root_horizontal()
        if 'root' in painting_features:
            plot_root()
        if 'global_joint' in painting_features:
            #assert guidance is not None
            #plot_global_joint(guidance['joint'], guidance['mask'])
            pass

        for feat in painting_features:
            plot_feature(feat)
        
        if pointcloud is not None:
            cloud = pointcloud[index]
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color="gray", s=1, alpha=0.5)
            num = index // 10
            # index = index_list["interact_contact"][num]
            # contact = cloud[index]
            # ax.scatter(contact[:, 0], contact[:, 1],contact[:, 2], color='red', s=10, alpha=0.5)
            # index = index_list["interact_far"][num]
            # far = cloud[index]
            # ax.scatter(far[:, 0], far[:, 1],far[:, 2], color='blue', s=10, alpha=0.5)

            

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 // fps, repeat=False)

    ani.save(save_path, fps=fps)

    plt.close()




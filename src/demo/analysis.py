from enum import Enum
import logging
import cv2

from matplotlib import pyplot as plt
from scene_classification import SceneClass
import numpy as np

class Possession(Enum):
    TEAM_A = 0
    TEAM_B = 1


class TerritoryAnalysis: 
    xedges = None
    yedges = None
    def __init__(self, xbin_edges = [200, 360, 520, 723, 925, 1128, 1330, 1490, 1650], ybin_edges = [43, 296.25, 549.5, 802.75, 1056]): 
        self.xbin_edges = xbin_edges
        self.ybin_edges = ybin_edges
        self.territory_hist = np.zeros((len(self.xbin_edges)-1, len(self.ybin_edges)-1))
        self.weighted_territory_hist = np.zeros((len(self.xbin_edges)-1, len(self.ybin_edges)-1))
    
    def analyse_frame(self, player_positions, scene_class): 
        if scene_class == SceneClass.WIDE_VIEW or scene_class == SceneClass.CLOSE_VIEW: 
            hist, self.xedges, self.yedges = np.histogram2d([x for x, _ in player_positions], [y for _, y in player_positions], bins=(self.xbin_edges, self.ybin_edges), range=[[200, 1650], [43, 1056]])
            
            self.weighted_territory_hist += self._weight_detections_naive(hist, len(player_positions))
            self.territory_hist += hist
    
    def _weight_detections_naive(self, hist, num_players): 
        return hist * (min(15, num_players) / 15)
    
    def _display_graphs(self):
        fig, axs = plt.subplots(3, 2, figsize=(11, 11))
        fig.set_dpi(300)
        plt.tight_layout(pad=3)
        plt.margins(x = 0.1, y = 3)
        cols = ['{}'.format(col) for col in ["Territory heatmap", "Weighted territory heatmap"]]
        rows = ['{} bins'.format(row) for row in ['32', '4', '2']]
        for ax, col in zip(axs[0], cols):
            ax.annotate(col, xy=(0.5, 1.1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=15, ha='center', va='baseline')

        for ax, row in zip(axs[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        fontsize=12, ha='right', va='center')
            
        plt.setp(axs.flat, xlabel='Distance from goal line (m)', ylabel='England')

        for i, hist in enumerate([self.territory_hist/self.territory_hist.sum()*100, self.weighted_territory_hist/self.weighted_territory_hist.sum()*100]):
            xtick_loc = np.array([200, 520, 925, 1330, 1650])
            xtick_label = ["0", "22", "50", "22", "0"]

            H = hist
            X, Y = np.meshgrid(self.xbin_edges, self.ybin_edges)
            pcm = axs[0][i].pcolormesh(X, Y, H.T)
            ax1 = axs[0][i].twinx()
            ax1.set_ylabel("New Zealand", rotation=-90, labelpad=10)
            ax1.set_yticklabels([])
            axs[0][i].set_xticks(xtick_loc) 
            axs[0][i].set_xticklabels(xtick_label)
            axs[0][i].set_ylim(axs[0][i].get_ylim()[::-1])
            axs[0][i].set_yticklabels([])
            # axs[0][i].set_xlabel("Distance from goal line (m)")
            self._add_text(H, axs[0][i], self.xbin_edges, self.ybin_edges)
            axs[0][i].grid(False)

            H = self.bin_ndarray(hist, (4, 1))
            X, Y = np.meshgrid(self.xbin_edges[::2], np.array([self.ybin_edges[0], self.ybin_edges[-1]]))

            pcm = axs[1][i].pcolormesh(X, Y, H.T)
            ax1 = axs[1][i].twinx()
            ax1.set_ylabel("Ireland", rotation=-90, labelpad=10)
            ax1.set_yticklabels([])
            pcm = axs[1][i].pcolormesh(X, Y, H.T)
            axs[1][i].set_xticks(xtick_loc) 
            axs[1][i].set_xticklabels(xtick_label)
            axs[1][i].set_ylim(axs[1][i].get_ylim()[::-1])
            # axs[1][i].set_xlabel("Distance from goal line (m)")
            axs[1][i].set_yticklabels([])
            self._add_text(H, axs[1][i], self.xbin_edges[::2], np.array([self.ybin_edges[0], self.ybin_edges[-1]]))
            axs[1][i].grid(False)

            H = self.bin_ndarray(hist, (2, 1))
            X, Y = np.meshgrid(self.xbin_edges[::4], np.array([self.ybin_edges[0], self.ybin_edges[-1]]))

            pcm = axs[2][i].pcolormesh(X, Y, H.T)
            ax1 = axs[2][i].twinx()
            ax1.set_ylabel("Ireland", rotation=-90, labelpad=10)
            ax1.set_yticklabels([])
            pcm = axs[2][i].pcolormesh(X, Y, H.T)
            axs[2][i].set_xticks(xtick_loc) 
            axs[2][i].set_xticklabels(xtick_label)
            axs[2][i].set_yticklabels([])
            # axs[2][i].set_xlabel("Distance from goal line (m)")
            self._add_text(H, axs[2][i], self.xbin_edges[::4], np.array([self.ybin_edges[0], self.ybin_edges[-1]]))
            axs[2][i].set_ylim(axs[2][i].get_ylim()[::-1])
            axs[2][i].grid(False)
        fig.savefig("eng_nzl_terr_map.png")
    def _add_text(self, hist, ax, xedges, yedges): 
        for i in range(1, len(xedges)): 
            for j in range(1, len(yedges)): 
                xcenter = (xedges[i] + xedges[i-1]) / 2
                ycenter = (yedges[j] + yedges[j-1]) / 2
                ax.text(xcenter, ycenter, f"{hist[i-1, j-1]:.2f}%", ha="center", va="center")
    
    def teamA_territory(self, weighted): 
        return self.territory_hist if not weighted else self.weighted_territory_hist
    

    def bin_ndarray(self, ndarray, new_shape, operation='sum'):

        operation = operation.lower()
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                            new_shape))
        compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                    ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))
        return ndarray


class PossessionAnalysis: 
    total_frames = 0
    num_close_frames = 0
    num_close_teamA_frames = 0

    def __init__(self, team_color): 
        self.previous_possession = None
        self.num_teamA_frames = 0
        self.teamA_color = team_color
        self.previous_scene = None
        self.possession = None
        

    def _bounding_box_center(self, bounding_boxes): 
        return np.array([[(x2 + x1)/2, (y2 + y1)/2] for x1, y1, x2, y2 in bounding_boxes])

    def _closest_player_detection(self, ball_detections, player_detections): 
        player_detection_centers = self._bounding_box_center(player_detections)
        ball_center = self._bounding_box_center(ball_detections)[0] #Get first ball detection (gaurenteed to be highest conf by detectron)
        return np.argmin(np.array([pow(player_center[0] - ball_center[0], 2) + pow(player_center[1] - ball_center[1], 2) for player_center in player_detection_centers]))
    

    def analyse_frame(self, ball_detections, player_detections, jersey_detections, scene_class, vis_frame=None):
        if scene_class == SceneClass.CLOSE_VIEW:
            if len(ball_detections) == 0 or len(player_detections) == 0: 
                self.previous_scene = scene_class
                return vis_frame

            player_idx = self._closest_player_detection(ball_detections, player_detections)
            vis_frame = cv2.line(vis_frame, (self._bounding_box_center(ball_detections)[0]).astype(int), (self._bounding_box_center(player_detections)[player_idx]).astype(int), (0, 0, 255))

            if (jersey_detections[0][player_idx] == self.teamA_color).all(): 
                self.num_close_frames += 1
                self.num_close_teamA_frames += 1
                self.previous_scene = scene_class
            else: 
                self.num_close_frames += 1
                self.previous_scene = scene_class

            return vis_frame
        if scene_class == SceneClass.WIDE_VIEW:

            if self.previous_scene == SceneClass.CLOSE_VIEW:
                if self.num_close_frames < 2: 
                    pass
                elif self.num_close_teamA_frames / self.num_close_frames > 0.5: 
                    logging.info("Team A has possession")
                    self.possession = Possession.TEAM_A
                    self.num_teamA_frames += self.num_close_frames
                    self.total_frames += self.num_close_frames
                else: 
                    logging.info("Team B has possession")
                    self.possession = Possession.TEAM_B
                self.num_close_frames = 0
                self.num_close_teamA_frames = 0
                self.previous_scene = scene_class
            
            if self.possession == None: 
                return vis_frame

            self.total_frames += 1

            if self.possession == Possession.TEAM_A: 
                self.num_teamA_frames += 1
                self.previous_scene = scene_class
        return vis_frame

    def possession_stats(self):
        if self.total_frames == 0:
            return 0
        return self.num_teamA_frames,  self.total_frames

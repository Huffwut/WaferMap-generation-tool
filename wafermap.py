import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
import math

class Wafermap:

    def __init__(self, dataframe):
        """Load dataframe file with columns representing different measurements. Must have X and Y as the last columns.
        Args:
            pandas DataFrame
        """
        self.__dataframe = dataframe
        self.__dataframe_wo_outliers = None # imo the software should only be used for cleaned data without any outliers
        self.__actual_pixel_size = 20

        self.__background_pixel = self.__background()
        self._outlier_pixel = self.__outlier_pixel()

        self.__wafer_height = 0
        self.__wafer_width = 0
        self.__wafer_size = int((max(self.sizeof_wafer())+5) * 2)

        self.__stored_image = None
        self._absolute_min = None
        self.__outlier_index = None
        self.__markings = None

    def set_outliers(self, indexes):
        self.__dataframe_wo_outliers = self.__dataframe.drop(self.__dataframe.reindex(indexes).index, axis=0)
        self.__outlier_index = indexes
        # should be able to pass column, and define the range
        # pass

    def set___markings(self, indexes):
        self.__markings = indexes

    def sizeof_wafer(self):
        """Find out approximately what size is the wafermap

        Returns:
            (lowest possible coodinate, highest possible coordinate)
        """
        find_lowest = self.__dataframe[self.__dataframe.columns[-2:]].min()
        find_highest = self.__dataframe[self.__dataframe.columns[-2:]].max()
        return abs(min(find_lowest)), abs(max(find_highest))

    def __pixel(self, val):
        if self.__actual_pixel_size < 3:
            raise Exception('pixel self.__wafer_width and self.__wafer_height should not be smaller than 3. Current set value is', self.__actual_pixel_size)
        x = np.full((int(self.__actual_pixel_size),int(self.__actual_pixel_size)), self._absolute_min-1 ,dtype=float) #20x20 pixel block size, surrounding values will be black due to lowest possible number
        x[1:-1,1:-1] = val
        return x

    def __marking_pixel(self, val):
        x = np.full((int(self.__actual_pixel_size),int(self.__actual_pixel_size)), self._absolute_min-1 ,dtype=float) #20x20 pixel block size, surrounding values will be black due to lowest possible number
        x[6:-6,6:-6] = val
        return x

    def boxplot(self):
        print('not implemented yet')
        # depends on how many columns there are...
        # f, axes = plt.subplots(2,3, figsize=(18,11))
        # names = result.columns.values.tolist()
        # count = 0
        # for i in range(2):
        #     for j in range(3):
        #         finished_data.boxplot([names[count]], ax=axes[i][j])
        #         count+=1
        # f.suptitle('Parameters')
        # # f.text(0.1, 0.05, text)
        # plt.show()
        pass

    def _abs_min(self):
        """returns lowest possible number from the dataframe
        """
        return min(self.__dataframe.min()) # as long as it is the lowest value, it doesnt matter from which column/row

    def _abs_max(self):
        """returns highest possible number from dataframe
        """
        return max(self.__dataframe.max())

    def __background(self):
        missing_defective = np.zeros((self.__actual_pixel_size, self.__actual_pixel_size), int)
        np.matrix.fill(missing_defective, int(self._abs_min()-10))
        np.fill_diagonal(missing_defective, int(self._abs_max()+10))
        reverse_m = np.fliplr(missing_defective)

        for row in range(len(reverse_m)):
            for col in range (len(reverse_m)):
                if reverse_m[row][col] != int(self._abs_min()-10):
                    missing_defective[row][col] = reverse_m[row][col]

        return missing_defective

    def __outlier_pixel(self):
        defective = np.zeros((self.__actual_pixel_size, self.__actual_pixel_size), int)
        np.matrix.fill(defective, int(self._abs_max()+10)) # outliers have their own color, so its ok
        np.fill_diagonal(defective, int(self._abs_min()-10))
        reverse_m = np.fliplr(defective)

        for row in range(len(reverse_m)):
            for col in range(len(reverse_m)):
                if reverse_m[row][col] != int(self._abs_max()+10):
                    defective[row][col] = reverse_m[row][col]

        return defective

    def reset_background_values(self):
        """If there were any changes within the dataset and the background doesn't make sense anymore... this should fix it
        """
        self.__background_pixel = self.__background()
        self._outlier_pixel = self.__outlier_pixel()
    
    def attempt_image(self):
        '''
        Attempts to create wafermap image internally. Use plot_wafer() after this is complete.
        '''
        middle_x = round(self.__wafer_size/2)
        middle_y = round(self.__wafer_size/2)
        n = int(self.__wafer_size)
        m = int(self.__wafer_size)
        _backgrnd = self.__background_pixel
        self._absolute_min = self._abs_min()


        if self.__dataframe_wo_outliers is not None:
            df = copy.deepcopy(self.__dataframe_wo_outliers)
            # df = self.__dataframe_wo_outliers.copy()
            defective = self.__outlier_pixel()

        else:
            df = copy.deepcopy(self.__dataframe)
            # df = self.__dataframe.copy()

        X_coord = np.asarray(df[df.columns[-2]]).reshape(-1,1)
        Y_coord = np.asarray(df[df.columns[-1]]).reshape(-1,1)
        each_wafer = {}

        for each_col in df.columns[:-2]: # this becomes for each wafermap
            i = 0  
            tmp = np.asarray(df[each_col]).reshape(-1,1)
            scaled = tmp
            image = [[0 for b in range(m)] for d in range(n)]
        
            for row in range(n):
                for col in range(m):
                    image[row][col] = _backgrnd

            for each_data in scaled:
                image[int(middle_x+Y_coord[i])][int(middle_y+X_coord[i])] = self.__pixel(*each_data)
                i = i + 1
       
            each_wafer[each_col] = image

        if self.__dataframe_wo_outliers is not None:
            X_coord = np.asarray(self.__dataframe.reindex(self.__outlier_index)['X']).reshape(-1,1)
            Y_coord = np.asarray(self.__dataframe.reindex(self.__outlier_index)['Y']).reshape(-1,1)

            for each_col in self.__dataframe.columns[:-2]:
                copy_img = copy.deepcopy(each_wafer[each_col])

                for i in range(len(self.__outlier_index)):
                    copy_img[int(middle_x+Y_coord[i])][int(middle_y+X_coord[i])] = defective

                each_wafer[each_col] = copy_img

        self.__stored_image = each_wafer
        # return each_wafer

    def attempt_image_block(self):
        df_size = len(self.__dataframe)

        self.__wafer_width = int(np.sqrt(df_size))
        self.__wafer_height = int(np.sqrt(df_size))

        while (self.__wafer_width * self.__wafer_height < df_size):
            self.__wafer_width = self.__wafer_width + 1

        _backgrnd = self.__background_pixel
        self._absolute_min = self._abs_min()

        if self.__dataframe_wo_outliers is not None:
            df = self.__dataframe_wo_outliers.copy()
            defective = self.__outlier_pixel()

        else:
            df = copy.deepcopy(self.__dataframe)

        each_wafer = {}

        for each_col in df.columns: # this becomes for each wafermap
            i = 0  
            j = 0
            tmp = np.asarray(df[each_col]).reshape(-1,1)
            scaled = tmp
            image = [[0 for b in range(self.__wafer_width)] for d in range(self.__wafer_height)]
        
            for row in range(self.__wafer_height):
                for col in range(self.__wafer_width):
                    image[row][col] = _backgrnd

            for each_data in scaled:
                image[j][i] = self.__pixel(*each_data)
                i = i + 1
                if i % self.__wafer_width == 0:
                    j += 1
                    i = 0
       
            each_wafer[each_col] = image

        self.__stored_image = each_wafer 
    
    def mark_indexes(self):
        middle_x = round(self.__wafer_size/2)
        middle_y = round(self.__wafer_size/2)
        df = copy.deepcopy(self.__dataframe)
        # print(self.__markings)
        X_coord = np.asarray(self.__dataframe.loc[self.__markings]['X']).reshape(-1,1)
        Y_coord = np.asarray(self.__dataframe.loc[self.__markings]['Y']).reshape(-1,1)

        for each_col in df.columns[:-2]: # this becomes for each wafermap
            copy_img = copy.deepcopy(self.__stored_image[each_col])

            for i in range(len(X_coord)):
                copy_img[int(middle_x+Y_coord[i])][int(middle_y+X_coord[i])] = self.__marking_pixel(np.max(copy_img[int(middle_x+Y_coord[i])][int(middle_y+X_coord[i])]))
            
            self.__stored_image[each_col] = copy_img

    def plot_wafer(self, c_color='jet', font_size=22, cmap_min=None, cmap_max=None, save=None):
        c_cmap = plt.get_cmap(c_color)
        c_cmap.set_under('black')
        c_cmap.set_over('white')
        plt.rcParams.update({'font.size': font_size})

        if self.__dataframe_wo_outliers is not None:
            df = copy.deepcopy(self.__dataframe_wo_outliers)

        else:
            df = copy.deepcopy(self.__dataframe)

        # TODO: add custom color bar, meaning the units of the color bar, whether its resistance of mV

        for each_map in self.__stored_image.keys():
            im = np.asarray(self.__stored_image[each_map])
            are = im[0].reshape((self.__wafer_size * self.__actual_pixel_size), self.__actual_pixel_size).T
            for i in range(1, self.__wafer_size):
                are = np.vstack((are, im[i].reshape((self.__wafer_size * self.__actual_pixel_size), self.__actual_pixel_size).T))

            fig = plt.figure(figsize=(12,12))
            if cmap_min is not None:
                plt.imshow(are, cmap=c_cmap, vmin=cmap_min, vmax=cmap_max)
            else:
                plt.imshow(are, cmap=c_cmap, vmin=df[each_map].min(), vmax=df[each_map].max())

            plt.colorbar()
            plt.title(each_map)
            plt.xlabel('x px')
            plt.ylabel('y px')
            if save is not None:
                if not os.path.isdir(save):
                    os.makedirs(save)
                fig.savefig(save + '/' + each_map + '.svg')
            plt.show()

    def plot_mondriaan(self, c_color='jet', font_size=22, cmap_min=None, cmap_max=None, save=None):
        c_cmap = plt.get_cmap(c_color)
        c_cmap.set_under('black')
        c_cmap.set_over('white')
        plt.rcParams.update({'font.size': font_size})

        if self.__dataframe_wo_outliers is not None:
            df = copy.deepcopy(self.__dataframe_wo_outliers)

        else:
            df = copy.deepcopy(self.__dataframe)

        # TODO: add custom color bar, meaning the units of the color bar, whether its resistance of mV

        for each_map in self.__stored_image.keys():
            im = np.asarray(self.__stored_image[each_map])
            are = im[0].reshape((self.__wafer_width * self.__actual_pixel_size), self.__actual_pixel_size).T
            for i in range(1, self.__wafer_height):
                are = np.vstack((are, im[i].reshape((self.__wafer_width * self.__actual_pixel_size), self.__actual_pixel_size).T))

            fig = plt.figure(figsize=(12,12))
            if cmap_min is not None:
                plt.imshow(are, cmap=c_cmap, vmin=cmap_min, vmax=cmap_max)
            else:
                plt.imshow(are, cmap=c_cmap, vmin=df[each_map].min(), vmax=df[each_map].max())

            plt.colorbar()
            plt.title(each_map)
            plt.xlabel('x px')
            plt.ylabel('y px')
            if save is not None:
                if not os.path.isdir(save):
                    os.makedirs(save)
                fig.savefig(save + '/' + each_map + '.svg')
            plt.show()
clc; clear all; close all;
root = '/Users/fabianyii/Desktop/my_MMAC_annotation/';

%% MMAC dataset
img_path = fullfile(root, 'shuffled');
% output (masked images) saved to the following folder (for macular
% atrophy)
masked_img_path = root + "MA_masked_MMAC/"; 

% output (masked images) saved to the following folder (for patchy
% atrophy)
masked_img_path = root + "patchy_masked_MMAC/"; 

% get names of all images
sampled_img_file_path = fullfile(root, "annotation", "linkage.xls");
sampled_img_file = readtable(sampled_img_file_path); 
Nrows = size(sampled_img_file, 1);
empty = cell(Nrows,1);
sampled_img_file.fovea_x = empty;
sampled_img_file.fovea_y = empty;

i_MA = (sampled_img_file.GT_grade == 4);
i_patchy = (sampled_img_file.GT_grade == 3);
i_diffuse = (sampled_img_file.GT_grade == 2);

MA_names = sampled_img_file(i_MA,:).shuffledName;
patchy_names = sampled_img_file(i_patchy,:).shuffledName;
all_names = sampled_img_file.shuffledName;


%% PALM dataset
img_path = fullfile(root, 'PALM');
% output (masked images) saved to the following folder (for macular
% atrophy)
masked_img_path = root + "MA_masked_PALM/"; 

% % output (masked images) saved to the following folder (for patchy
% % atrophy)
% masked_img_path = root + "patchy_masked_PALM/"; 

% get names of all images
sampled_img_file_path = fullfile(root, "annotation", "myPALMannotation.csv");
sampled_img_file = readtable(sampled_img_file_path); 

i_MA = (sampled_img_file.MMcategory_second == 4);
i_patchy = (sampled_img_file.MMcategory_second == 3);
i_diffuse = (sampled_img_file.MMcategory_second == 2);

MA_names = sampled_img_file(i_MA,:).name;
patchy_names = sampled_img_file(i_patchy,:).name;
all_names = sampled_img_file.name;


%% read one image at a time so it can be loaded to the imageSegmenter app
i = 153
img_name = MA_names(i);
I = imread(fullfile(img_path, img_name{1}));
imageSegmenter(I);
clc; disp(img_name{1});


%% save 'maskedImage'
close all;
imwrite(BW, fullfile(masked_img_path,img_name{1}));
imshow(BW);
% clear from workspace so they don't clutter up the space!
clear('maskedImage'); clear('BW');



%% identify foveal coordinates
indices = find(i_diffuse == 1 | i_patchy == 1);
for i=20:length(i_diffuse)

    i
    row = indices(i);
    name = sampled_img_file(row, :).shuffledName{1};
    im_path = imread(fullfile(root, 'shuffled', name));
    imshow(im_path);

    pause;

    [x,y] = ginput(1);
    sampled_img_file(row, :).fovea_x{1} = x;
    sampled_img_file(row, :).fovea_y{1} = y;
    writetable(sampled_img_file, fullfile(root, 'annotation', 'myMMACannotationWithFoveaCoordinates.csv'));

end




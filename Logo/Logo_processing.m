%% read png
clear
clc
A = imread('NPSAT_blur.png');
%% write image as scattered interpolation file
Ny = size(A,1);
Nx = size(A,2);
B = double(A(1:Ny,1:Nx,1));
[Xgrid, Ygrid] = meshgrid(1:Nx, Ny:-1:1);
XX = reshape(Xgrid,Nx*Ny, 1);
YY = reshape(Ygrid,Nx*Ny, 1);
VV = reshape(B,Nx*Ny, 1);
%%
writeScatteredData('logo_data.npsat', struct('PDIM',2,'TYPE','FULL','MODE','SIMPLE'), [XX YY VV]);
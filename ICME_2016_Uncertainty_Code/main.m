%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compute Results for Spatial Correlation study of Eye-fixation maps as
%  proposed in "UNDERSTANDING SPATIAL CORRELATION IN EYE-FIXATION MAPS FOR
%  VISUAL ATTENTION IN VIDEOS" presented at ICME 2016, Seattle, Washigton.
%  Written by Tariq Alshawi, PhD student, Georgia Instituet of Technology
%  contact: talshawi@gatech.edu
%  Last update: 01/19/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              ****************   NOTE  *******************
%
%   The code is divided into three seperate sections to facilitate faster
%   run time and easier modification to code parameters. Section A,
%   produces results in Fig.2. Section B produces results in Fig.3(b),
%   Fig.4(b), and Fig.5(b). Section c produces results in Fig.6 and Fig.7
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               SECTION A
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%%% Specify lists for input files
iList={'beverly01.mpg'...   %1
    'beverly03.mpg'...      %2
    'beverly05.mpg'...      %3
    'beverly06.mpg'...      %4
    'beverly07.mpg'...      %5
    'beverly08.mpg'...      %6
    'gamecube02.mpg'...     %7
    'gamecube04.mpg'...     %8
    'gamecube05.mpg'...     %9
    'gamecube06.mpg'...     %10
    'gamecube13.mpg'...     %11
    'gamecube16.mpg'...     %12
    'gamecube17.mpg'...     %13
    'gamecube18.mpg'...     %14
    'gamecube23.mpg'...     %15
    'monica03.mpg'...       %16
    'monica04.mpg'...       %17
    'monica05.mpg'...       %18
    'monica06.mpg'...       %19
    'saccadetest.mpg'...    %20
    'standard01.mpg'...     %21
    'standard02.mpg'...     %22
    'standard03.mpg'...     %23
    'standard04.mpg'...     %24
    'standard05.mpg'...     %25
    'standard06.mpg'...     %26
    'standard07.mpg'...     %27
    'tv-action01.mpg'...    %28
    'tv-ads01.mpg'...       %29
    'tv-ads02.mpg'...       %30
    'tv-ads03.mpg'...       %31
    'tv-ads04.mpg'...       %32
    'tv-announce01.mpg'...  %33
    'tv-music01.mpg'...     %34
    'tv-news01.mpg'...      %35
    'tv-news02.mpg'...      %36
    'tv-news03.mpg'...      %37
    'tv-news04.mpg'...      %38
    'tv-news05.mpg'...      %39
    'tv-news06.mpg'...      %40
    'tv-news09.mpg'...      %41
    'tv-sports01.mpg'...    %42
    'tv-sports02.mpg'...    %43
    'tv-sports03.mpg'...    %44
    'tv-sports04.mpg'...    %45
    'tv-sports05.mpg'...    %46
    'tv-talk01.mpg'...      %47
    'tv-talk03.mpg'...      %48
    'tv-talk04.mpg'...      %49
    'tv-talk05.mpg'...      %50
    };

% intialize script variables
EntropyMtx = zeros(1,length(iList));
ConditionalEntropyMtx2 = zeros(1,length(iList));
ConditionalEntropyMtxRandom = zeros(1,length(iList));

pixel = zeros(12,16,length(iList));
pixel_avg2 = zeros(12,16,length(iList));


for k=1:length(iList)
    % load Eye-fixation map: 'subsampledFxTruth',
    % for video#k
    disp(['k= ' num2str(k)])
    nFName=strrep(iList{k}, '.mpg', '_subsampledFxTruth.mat');
    load([pwd '\EyeFixationMaps\' nFName]);
    
    for i=2:size(subsampledFxTruth,1)-1;
        for j=2:size(subsampledFxTruth,2)-1;
            for t=2:size(subsampledFxTruth,3)-1;
                % Extract value of pixel (i,j,t) into variable 'pixel'
                pixel(i,j,t)       = squeeze(subsampledFxTruth(i,j,t));
                % Extract vlue of pixel(i,j,t) neighbours into variable
                % 'neighbor'
                neighbor    = subsampledFxTruth(i-1:i+1,j-1:j+1,t-1:t+1);
                % eleminate value of current pixel (i,j,t) for average
                neighbor(2,2,2) = 0;
                % compute neighbourhood average
                pixel_avg2(i,j,t)   = mean(neighbor(:));
            end
        end
    end
    x = pixel(:)';
    y2 = pixel_avg2(:)';
    
    %%%%%%%%%%%% Eleminate 50% of the zeros in the eye-fixation map %%%%%%%%%%
    %%%%%% This results in shifting all the curves up = increase entropy %%%%%
    %     listZeros = find(~x);
    %     index = randn(size(listZeros))>0.1;
    %     aVector = listZeros.*index;
    %     aVector(aVector == 0) = [];
    %     x(aVector) = [];
    %     y2(aVector) = [];
    
    % Compute entropy of normalized version of variable x
    EntropyMtx(k) = entropy(x/max(x));
    % compute conditional entropy of pixel value 'x' given neighbours
    % average 'y2' using conditionalEntropy function(included in this
    % folder). Note that conditionalEntropy works on interger values only,
    % thus, 'y2' is rounded to nearest integer with 10^-2 acccuracy
    ConditionalEntropyMtx2(k) = conditionalEntropy(x,round(y2*10));
    % compute conditional entropy of pixel value 'x' given a random variabe
    ConditionalEntropyMtxRandom(k) = conditionalEntropy(x,round(rand(1,length(y2))*100));
end


figure;
plot(1:50,EntropyMtx,'-x',1:50,...
    ConditionalEntropyMtx2,'-s',1:50,ConditionalEntropyMtxRandom,'-d')
h=legend('Entropy (Eq.3)', ...
    'Conditional Entropy Given Neighborhood average (Eq.4)', ...
    'Conditional Entropy Given random variable');
set(h,'location','northeast')
xlabel('Video segemnt number')
ylabel('Entropy (bits)')


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               SECTION B
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc
%%% Specify lists for input files
iList={'beverly01.mpg'...   %1
    'beverly03.mpg'...      %2
    'beverly05.mpg'...      %3
    'beverly06.mpg'...      %4
    'beverly07.mpg'...      %5
    'beverly08.mpg'...      %6
    'gamecube02.mpg'...     %7
    'gamecube04.mpg'...     %8
    'gamecube05.mpg'...     %9
    'gamecube06.mpg'...     %10
    'gamecube13.mpg'...     %11
    'gamecube16.mpg'...     %12
    'gamecube17.mpg'...     %13
    'gamecube18.mpg'...     %14
    'gamecube23.mpg'...     %15
    'monica03.mpg'...       %16
    'monica04.mpg'...       %17
    'monica05.mpg'...       %18
    'monica06.mpg'...       %19
    'saccadetest.mpg'...    %20
    'standard01.mpg'...     %21
    'standard02.mpg'...     %22
    'standard03.mpg'...     %23
    'standard04.mpg'...     %24
    'standard05.mpg'...     %25
    'standard06.mpg'...     %26
    'standard07.mpg'...     %27
    'tv-action01.mpg'...    %28
    'tv-ads01.mpg'...       %29
    'tv-ads02.mpg'...       %30
    'tv-ads03.mpg'...       %31
    'tv-ads04.mpg'...       %32
    'tv-announce01.mpg'...  %33
    'tv-music01.mpg'...     %34
    'tv-news01.mpg'...      %35
    'tv-news02.mpg'...      %36
    'tv-news03.mpg'...      %37
    'tv-news04.mpg'...      %38
    'tv-news05.mpg'...      %39
    'tv-news06.mpg'...      %40
    'tv-news09.mpg'...      %41
    'tv-sports01.mpg'...    %42
    'tv-sports02.mpg'...    %43
    'tv-sports03.mpg'...    %44
    'tv-sports04.mpg'...    %45
    'tv-sports05.mpg'...    %46
    'tv-talk01.mpg'...      %47
    'tv-talk03.mpg'...      %48
    'tv-talk04.mpg'...      %49
    'tv-talk05.mpg'...      %50
    };

% intialize script variables
entropyMtx = zeros(12,16,length(iList));

% run for only three videos 10: gamecube06
%                           20: sccadedtest
%                           37: tv-news03
for k=[10 20 37]
    % load Eye-fixation map: 'subsampledFxTruth',
    % for video#k
    disp(['k= ' num2str(k)])
    nFName=strrep(iList{k}, '.mpg', '_subsampledFxTruth.mat');
    load([pwd '\EyeFixationMaps\' nFName]);
    for i=1:size(subsampledFxTruth,1);
        for j=1:size(subsampledFxTruth,2);
            % Extract pixel vector vlaues in location (i,j)
            pixel       = squeeze(subsampledFxTruth(i,j,:));
            % Extract pixel temporal neighbour
            pixel_neighbour  = circshift(pixel,1);
            % Compute mutual information between pixel (i,j) values
            % over time
            MI_pixel     = mutualInformation(pixel,pixel_neighbour);
            entropyMtx (i,j,k) = MI_pixel;
        end
    end
    L = squeeze(entropyMtx (:,:,k));
    figure;
    surf(L)
    colorbar
    view(2)
    xlabel('x-location (pixels)')
    ylabel('y-location (pixels)')
    title(['Shift = 1 - Scale 1 -' iList{k}]);
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               SECTION c
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc

%%% Specify lists for input and output files
iList={'beverly01.mpg'...   %1
    'beverly03.mpg'...      %2
    'beverly05.mpg'...      %3
    'beverly06.mpg'...      %4
    'beverly07.mpg'...      %5
    'beverly08.mpg'...      %6
    'gamecube02.mpg'...     %7
    'gamecube04.mpg'...     %8
    'gamecube05.mpg'...     %9
    'gamecube06.mpg'...     %10
    'gamecube13.mpg'...     %11
    'gamecube16.mpg'...     %12
    'gamecube17.mpg'...     %13
    'gamecube18.mpg'...     %14
    'gamecube23.mpg'...     %15
    'monica03.mpg'...       %16
    'monica04.mpg'...       %17
    'monica05.mpg'...       %18
    'monica06.mpg'...       %19
    'saccadetest.mpg'...    %20
    'standard01.mpg'...     %21
    'standard02.mpg'...     %22
    'standard03.mpg'...     %23
    'standard04.mpg'...     %24
    'standard05.mpg'...     %25
    'standard06.mpg'...     %26
    'standard07.mpg'...     %27
    'tv-action01.mpg'...    %28
    'tv-ads01.mpg'...       %29
    'tv-ads02.mpg'...       %30
    'tv-ads03.mpg'...       %31
    'tv-ads04.mpg'...       %32
    'tv-announce01.mpg'...  %33
    'tv-music01.mpg'...     %34
    'tv-news01.mpg'...      %35
    'tv-news02.mpg'...      %36
    'tv-news03.mpg'...      %37
    'tv-news04.mpg'...      %38
    'tv-news05.mpg'...      %39
    'tv-news06.mpg'...      %40
    'tv-news09.mpg'...      %41
    'tv-sports01.mpg'...    %42
    'tv-sports02.mpg'...    %43
    'tv-sports03.mpg'...    %44
    'tv-sports04.mpg'...    %45
    'tv-sports05.mpg'...    %46
    'tv-talk01.mpg'...      %47
    'tv-talk03.mpg'...      %48
    'tv-talk04.mpg'...      %49
    'tv-talk05.mpg'...      %50
    };


% intialize script variables
Lm=30;
MImtx2Temporal = zeros(3000,length(iList));
MImtx3Temporal = zeros(3000,length(iList));
MImtxRandomTemporal = zeros(3000,length(iList));
MIcurve = zeros(Lm,50);


for k=1:length(iList)
    % load Eye-fixation map: 'subsampledFxTruth',
    % for video#k
    disp(['k= ' num2str(k)])
    nFName=strrep(iList{k}, '.mpg', '_subsampledFxTruth.mat');
    load([pwd '\EyeFixationMaps\' nFName]);
   
    
    pixelTemporal = [];
    pixelTemporal_avg2 = [];
    pixelTemporal_avg3 = [];
    pixelTemporal = zeros(size(subsampledFxTruth,3), 12*16);
    pixelTemporal_avg2 = zeros(size(pixelTemporal));
    pixelTemporal_avg3 = zeros(size(pixelTemporal));
    
    
    L(k) = size(subsampledFxTruth,3);
    % Change time horizon u from 1-Lm 
    for u=1:Lm
        for t=1+u:size(subsampledFxTruth,3)-u;
            % Extract frame t
            Temp                        = squeeze(subsampledFxTruth(:,:,t));
            % Reshape as 1D vector
            pixelTemporal(t,:)          = Temp(:);
            % Extract 2u+1 frames centered at t 
            neighborTemporal2            = subsampledFxTruth(:,:,t-u:t+u);
            % Eleminate frame t from average
            neighborTemporal2(:,:,u+1)  = 0;
            % Compute average neighbouring frame 
            Temp2                       = mean(neighborTemporal2,3);
            % Extract 2 frame at distance u from frame t
            neighborTemporal            = squeeze(subsampledFxTruth(:,:,t-u))+squeeze(subsampledFxTruth(:,:,t+u));
            % Compute average of distanced average frames 
            Temp3                       = neighborTemporal/2;
            pixelTemporal_avg2(t,:)     = Temp2(:);
            pixelTemporal_avg3(t,:)     = Temp3(:);
            
            
            xTemporal = squeeze(pixelTemporal(t,:));
            y2Temporal = squeeze(pixelTemporal_avg2(t,:));
            y3Temporal = squeeze(pixelTemporal_avg3(t,:));
            
            % Compute mutual information between frame t 'xTemporal' and average
            % neighbouring frame 'y2Temporal'
            MImtx2Temporal(t,k) = mutualInformation(xTemporal,round(y2Temporal*100));
            % Compute mutual information between frame t 'xTemporal' and average
            % distanced frame 'y3Temporal'
            MImtx3Temporal(t,k) = mutualInformation(xTemporal,round(y3Temporal*100));
            % Compute mutual information between frame t 'xTemporal' and
            % random variable
            MImtxRandomTemporal(t,k) = mutualInformation(xTemporal,round(randn(1,length(y2Temporal))*100));
            
        end
        % averaging
        MIcurve2(u,k) = mean(MImtx2Temporal(1:L(k),k));
        MIcurve3(u,k) = mean(MImtx3Temporal(1:L(k),k));
    end
    
end

%%%%%%%% Category-Based Localization %%%%%%%%%%%%
figure;
plot(1:Lm, mean(MIcurve2(:,1:6),2),'-*', ...
    1:Lm, mean(MIcurve2(:,7:15),2),'-s', ...
    1:Lm, mean(MIcurve2(:,16:19),2),'-d', ...
    1:Lm, MIcurve2(:,20),'-o', ...
    1:Lm, mean(MIcurve2(:,21:27),2),'-x', ...
    1:Lm, MIcurve2(:,28),'-<', ...
    1:Lm, mean(MIcurve2(:,29:32),2),'->', ...
    1:Lm, MIcurve2(:,33),'-p', ...
    1:Lm, MIcurve2(:,34),'-o', ...
    1:Lm, mean(MIcurve2(:,35:41),2),'-d', ...
    1:Lm, mean(MIcurve2(:,42:46),2),'-s', ...
    1:Lm, mean(MIcurve2(:,47:50),2),'-*')
xlim([0 20])
xlabel('Frame Distance')
ylabel('Mutual Information (bits)')
h=legend('Beverly','GameCube','Monica','Saccadetest','Standard',...
    'Tv-action','Tv-ads','Tv-announce','Tv-music','Tv-news',...
    'Tv-sports','Tv-talk','Reference');
set(h,'location','northeast')
title('Category-Based Average Information')


%%%%%%%% Category-Based Average information %%%%%%%%%%%%
figure;
plot(1:Lm, mean(MIcurve3(:,1:6),2),'-*', ...
    1:Lm, mean(MIcurve3(:,7:15),2),'-s', ...
    1:Lm, mean(MIcurve3(:,16:19),2),'-d', ...
    1:Lm, MIcurve3(:,20),'-o', ...
    1:Lm, mean(MIcurve3(:,21:27),2),'-x', ...
    1:Lm, MIcurve3(:,28),'-<', ...
    1:Lm, mean(MIcurve3(:,29:32),2),'->', ...
    1:Lm, MIcurve3(:,33),'-p', ...
    1:Lm, MIcurve3(:,34),'-o', ...
    1:Lm, mean(MIcurve3(:,35:41),2),'-d', ...
    1:Lm, mean(MIcurve3(:,42:46),2),'-s', ...
    1:Lm, mean(MIcurve3(:,47:50),2),'-*')
xlabel('Frame Distance')
ylabel('Mutual Information (bits)')
h=legend('Beverly','GameCube','Monica','Saccadetest','Standard',...
    'Tv-action','Tv-ads','Tv-announce','Tv-music','Tv-news',...
    'Tv-sports','Tv-talk','Reference');
set(h,'location','northeast')
title('Category-based Localization')
clear
close all
clc

% 2025-01-07 제작

%% ===================================== 데이터 불러오기 =======================================
[dat_file,dat_file_path] = uigetfile('*.txt','Select .dat file for TF analyzer data');

file = strcat(dat_file_path, dat_file);
        
fid = fopen(file);


buffer = fgetl(fid);

data = textscan(fid, '%f%f%f', 'Delimiter', '    ');

time = data{1,1};
ch1 = data{1,2};
ch2 = data{1,3};
% 
% figure(1)
% plot(time, ch1)

DefInput = 1/mean(gradient(time));


%% ===================================== condition input 및 error handle ================================================

[para1, para2, para3, para4, para5, filePath] = NCDialog_v2;

state = 0;
while state >=0
    if state == 0
        para2 = str2double(para2);
        para3 = str2double(para3);
        para4 = str2double(para4);
        para5 = str2double(para5);
        state = 1;
    end

    % 에러핸들
    if state ==1

        state = -1;

        if para2 < 0 , state = 100;  end
        if para3 <= 0 , state = 100;  end
        if para3 >= 100 , state = 100;  end
        if para4 <= 0 , state = 100;  end
        if para5 <= 0 , state = 100;  end

    end

    if state == 100,   error('Program aborted. 올바른 condition 값을 입력했는지 확인하세요'),    end
end

NCDialogPara.BiasSign = para1;
NCDialogPara.CapacitorArea = num2str(para2);      % 단위 : [um2]
NCDialogPara.QresTolerance = num2str(para3);          % 단위 : [%]
NCDialogPara.PulseWidth = num2str(para4);           % 단위 : [ns]
NCDialogPara.PulseRiseFall = num2str(para5);        % 단위 : [ns]

save(filePath, "NCDialogPara")


prompt1 = para1;
cap_area = para2 * 1e-12;

% if length(time) < 10000
%     newtime = linspace(time(1), time(end), 4*length(time)-3);
%     ch1 = interp1(time, ch1, newtime)';
%     time = newtime';
% 
%     ch1 = ch1 + max(ch1)/40 * rand(length(ch1),1);
% end


%% ===================================== 원본 current data 노이즈 제거하기 (FFT) =======================================



% -------------------------- FFT로 변환하여 주파수 도메인에서 살펴보기 ------------------------------
y_FFTout = fft(ch1);
% y2_FFTout = fft(ch2);

Fs = DefInput;
N = length(ch1);              % Get the Number of Points
k = 0:N-1;                  % Create a Vector from 0 to N-1
T = N/Fs;                   % Get the Minimum Frequency
freq = k/T;                 % Create a Vector fot the Frequency Range


% -------------------------- FFT로 변환하여 주파수 도메인에서 살펴보기 ------------------------------


PlotSizeRef = get(0,'screensize');

% fig1 = figure(1);
% fig1.Position = [10 50 0.7*PlotSizeRef(3) 0.7*PlotSizeRef(4)];
% subplot(2,1,1)
% plot(freq(2:end), abs(y_FFTout(2:end)))
% subplot(2,1,2)
% semilogx(freq(2:end), abs(y_FFTout(2:end)))



FreqLast = log10(freq(end)/2);     FreqInit = log10(freq(2));       FreqSearchRange = linspace(FreqInit, FreqLast, 60)';
%FreqLast - 2.2;

MSE = zeros(length(FreqSearchRange) , 1);
for i = 1:length(FreqSearchRange)-1

    filter_freq2double = 10^(FreqSearchRange(i));

    state = 1;
    while state >= 0

        if state == 1
            filter_freq_pos = find(freq > filter_freq2double, 1, 'first');

            final_FFTout = y_FFTout;
            final_FFTout(filter_freq_pos:length(y_FFTout) - filter_freq_pos + 1) = 0;

            ch1_return = ifft(final_FFTout);
            ch1_return = real(ch1_return);

            state = -1;
        end
    end
    MSE(i) = immse(ch1 , ch1_return);
end

errorChangeRate = gradient(FreqSearchRange) ./ gradient(log10(MSE));








MSE2 = zeros(length(FreqSearchRange) , 1);
for i = 1:length(FreqSearchRange)-1

    filter_freq2double2 = 10^(FreqSearchRange(i));

    state = 1;
    while state >= 0

        if state == 1
            filter_freq_pos = find(freq <= filter_freq2double2, 1, 'last');

            final_FFTout = y_FFTout;
            final_FFTout(1:filter_freq_pos) = 0;

            ch1_return = ifft(final_FFTout);
            ch1_return = real(ch1_return);

            state = -1;
        end
    end
    MSE2(i) = immse(ch1 , ch1_return);
end

mse_mean = mean(MSE2);             % MSE2의 평균값
[~, idx_closest] = min(abs(MSE2 - mse_mean));  % 평균값과의 차이가 최소가 되는 인덱스
% mse_func = @(c) mean((MSE2 - c).^2);  % MSE between f(x) and constant c
% 
% % Use fminbnd to minimize MSE over reasonable c-range (e.g., [-1, 1])
% [c_min, mse_min] = fminbnd(mse_func, FreqSearchRange(1), FreqSearchRange(end));

figure(2)
loglog(10.^FreqSearchRange , MSE) , grid on , grid minor, hold on
loglog(10.^FreqSearchRange , MSE2) , hold on
semilogx(10.^FreqSearchRange , (mean(MSE2)) * ones(1,length(FreqSearchRange)))




figure(3)
loglog(10.^FreqSearchRange , errorChangeRate)
title(strcat(dat_file_path,dat_file))

CutOffFreqIndex = find(errorChangeRate == min(errorChangeRate(idx_closest:end-1)));
CutOffFreq = 10^FreqSearchRange(CutOffFreqIndex)  *  1.15;


filter_freq_pos = find(freq > CutOffFreq, 1, 'first');

final_FFTout = y_FFTout;
final_FFTout(filter_freq_pos:length(y_FFTout) - filter_freq_pos + 1) = 0;

ch1_return = ifft(final_FFTout);
ch1_return = real(ch1_return);

% final_FFTout2 = y2_FFTout;
% final_FFTout2(filter_freq_pos:length(y2_FFTout) - filter_freq_pos + 1) = 0;
% 
% ch2_return = ifft(final_FFTout2);
% ch2_return = real(ch2_return);
% 
figure(4)
fig4 = figure(4);
plot(time, ch1, 'k-'), hold on
% plot(time , ch2_return), hold on
plot(time, ch1_return, 'r-', "LineWidth",2), grid on
fig4.Position = [0.3*PlotSizeRef(3) 50 0.5*PlotSizeRef(3) 0.7*PlotSizeRef(4)];


recover = ifft(y_FFTout);

%==================================================================================================================


tap = min(ch1(1:round(0.1*length(time)))) * 1.1;        ErrorSearchRange = linspace(-tap , tap , 200);

OSCMSE = zeros(length(ErrorSearchRange) , 1);
for j = 1:length(ErrorSearchRange)
    
    OSCMSE(j) = immse(ch1_return , ErrorSearchRange(j) * ones(length(ch1_return),1)) ;

end


% figure(5)
% plot(ErrorSearchRange, OSCMSE)

OSCErrorMean = ErrorSearchRange(find(OSCMSE == min(OSCMSE)));

% 
% % close(figure(4))
% figure(7)
% fig7 = figure(7);
% plot(time, ch1, 'k-'), hold on
% plot(time , OSCErrorMean * ones(length(time),1)), hold on
% plot(time, ch1_return, 'r-', "LineWidth",2), grid on
% fig7.Position = [0.3*PlotSizeRef(3) 50 0.5*PlotSizeRef(3) 0.7*PlotSizeRef(4)];

ch1_return_beforeAdj = ch1_return;
ch1_return = ch1_return - OSCErrorMean * ones(length(time),1);



% ----------------------------- Voltage 데이터 current로 바꾸기 --------------------------------
if strcmp(prompt1,'Negative biased') == 1
    ch1_return = -ch1_return;
end

current_data = ch1_return / 50 ;                    
% 50옴 채널에 VOLTAGE파가 들어오므로, V=IR에 의해서 50으로 나눠줘야 current level이 나옴
del_time = time(2:end) - time(1:end-1);
del_charge = del_time .* current_data(1:end-1);     % charge = del_time * current
accum_charge = cumsum(del_charge);                  % Calculating the accumulated sum
% ----------------------------- Voltage 데이터 current로 바꾸기 --------------------------------

% prompt1 = questdlg('Pulse sign을 고르세요', 'Dialog 1', 'Positive biased', 'Negative biased' , 'Positive biased');

% prompt3 = {'Capacitor Area를 입력하세요 (단위 : Wum^2):'};
% dlgtitle = 'Input';
% dims = [1 35];
% definput = {'10000'};
% opt.WindowStyle = 'normal';
% area = inputdlg(prompt3,dlgtitle,dims,definput,opt);
% 
% cap_area = str2double(cell2mat(area)) * 1e-12;
net_accum_charge_density = accum_charge / cap_area ;


%% ================================== Find Qchar , Qdis , Qres =======================================


[Q_charge, Q_charge_position] = max(real(net_accum_charge_density));
% Qcharge 위치 먼저 확정

slopeRange2 = length(  time(Q_charge_position: Q_charge_position+ round(0.5*(length(time)-Q_charge_position)))   );
%Qcharge 위치부터 맨 끝의 절반까지 탐색범위 설정
slopeFinder2 = zeros(slopeRange2 , 1);

for i = 1:slopeRange2-1
    slopeFinder2(i) = (net_accum_charge_density(end) - net_accum_charge_density(i+Q_charge_position-1))...
        / (time(end) - time(i+Q_charge_position-1)) ;
end

MaxSlope2 = find(slopeFinder2 == min(slopeFinder2)) + Q_charge_position;

Qres_position = find(time > time(MaxSlope2) + para5*1e-9 + 0.6*para4*1e-9 , 1 , "first");
Q_res = real(min(net_accum_charge_density(Qres_position:end)));
Qres_position = find(net_accum_charge_density == Q_res);
%Q_res = min(real(net_accum_charge_density(Q_charge_position:end)));
Q_discharge = Q_charge - Q_res ;



%% ========================================= Qres Handle ==========================================

slopeRange = floor(MaxSlope2 * 0.7);
slopeFinder = zeros(slopeRange,1);



for i = 1:slopeRange
    slopeFinder(i) = (net_accum_charge_density(MaxSlope2) - net_accum_charge_density(i)) / (time(MaxSlope2) - time(i)) ;
end

MaxSlope = find(slopeFinder == max(slopeFinder));
MaxSlope = round(0.75*MaxSlope);

polyfit_x = time(1:MaxSlope);           polyfit_y = net_accum_charge_density(1:MaxSlope);
Linregress = polyfit(polyfit_x, polyfit_y , 1);

QresAdjust = net_accum_charge_density - (time(1:end-1)*Linregress(1) + Linregress(2));
NewQ_charge = QresAdjust(Q_charge_position);
NewQ_res = min(real(QresAdjust(Qres_position:end)));
NewQ_discharge = NewQ_charge - NewQ_res;

ErrorRatio = 100 * NewQ_res / NewQ_charge;

%
% figure(5)
% plot(time(1:end-1), QresAdjust) , grid on, grid minor




if strcmp(prompt1,'Negative biased') == 1
    net_accum_charge_density = - net_accum_charge_density;

    Q_charge = - Q_charge;
    Q_discharge = - Q_discharge; 
    Q_res = - Q_res;

    NewQ_charge = - NewQ_charge;
    NewQ_discharge = - NewQ_discharge;
    NewQ_res = - NewQ_res;
end


%% ========================================= 후처리 ================================================

state = 1;
while state > 0

    if state == 1  % 원래 원본으로 먼저 그리기



        Q_charge_indic = strcat('Q_{charge} = ', num2str(Q_charge), ' [C/m^2]');
        Q_discharge_indic = strcat('Q_{discharge} = ', num2str(Q_discharge), ' [C/m^2]');
        Q_res_indic = strcat('Q_{res} = ', num2str(Q_res), ' [C/m^2]');
        Area_indic = strcat('Area_{cap} = ', num2str(cap_area * 1e12), '[\mum^2]');
        ErrorRate_indic = strcat('Qres size = ', num2str(ErrorRatio), '% of Q_{charge}');


        MaxFreq_indic = strcat('Max FFT freq. = ', num2str(DefInput));
        FreqIncrement_indic = strcat('FFT freq. inc. = ' , num2str(1/T));
        CutOffFreq_indic = strcat('CutOff freq. = ' , num2str(CutOffFreq));


        if strcmp(prompt1,'Positive biased') == 1
            fig6 = figure(6);
            fig6.Position = [10 50 0.95*PlotSizeRef(3) 0.8*PlotSizeRef(4)];
            subplot(1,2,1)
            plot(time, ch1, 'k-'), hold on
            plot(time , OSCErrorMean * ones(length(time),1)), hold on
            plot(time, ch1_return_beforeAdj, 'r-', "LineWidth",2), grid on, grid minor , hold off
            text(0.5*time(end), max(ch1), 'FFT info.','FontSize',12)
            text(0.5*time(end), 0.9*max(ch1), MaxFreq_indic,'FontSize',10)
            text(0.5*time(end), 0.8*max(ch1), FreqIncrement_indic,'FontSize',10)
            text(0.5*time(end), 0.7*max(ch1), CutOffFreq_indic,'FontSize',10)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('Current * 50\Omega [V]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'OSC 측정 profile, 노이즈 제거버전';' '},'FontSize',15)
            subplot(1,2,2)
            plot(time(1:end-1),net_accum_charge_density, 'k-'), grid on, grid minor , hold on
            plot(time(Q_charge_position), net_accum_charge_density(Q_charge_position), 'bo'), hold on
            plot(time(Qres_position), net_accum_charge_density(Qres_position), 'bo'), hold off
            if ErrorRatio > 0.5*para3 , text(0.6*time(end), 0.6*Q_charge, ErrorRate_indic,'FontSize',12), end
            text(0.6*time(end), 0.7*Q_charge, Q_res_indic,'FontSize',12)
            text(0.6*time(end), 0.8*Q_charge, Q_discharge_indic,'FontSize',12)
            text(0.6*time(end), 0.9*Q_charge, Q_charge_indic,'FontSize',12)
            text(0.6*time(end), Q_charge, Area_indic,'FontSize',12)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('ChargeDensity [C/m^2]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'정전 및 방전용량';' '},'FontSize',15)
            main_title = sgtitle(dat_file, 'Color', 'red','FontSize',20);

        elseif strcmp(prompt1,'Negative biased') == 1
            fig6 = figure(6);
            fig6.Position = [10 50 0.95*PlotSizeRef(3) 0.8*PlotSizeRef(4)];
            subplot(1,2,1)
            plot(time, ch1, 'k-'), hold on
            plot(time , OSCErrorMean * ones(length(time),1)), hold on
            plot(time, ch1_return_beforeAdj, 'r-', "LineWidth",2), grid on, grid minor , hold off
            text(0.1*time(end), max(ch1), 'FFT info.','FontSize',12)
            text(0.1*time(end), 0.9*max(ch1), MaxFreq_indic,'FontSize',10)
            text(0.1*time(end), 0.8*max(ch1), FreqIncrement_indic,'FontSize',10)
            text(0.1*time(end), 0.7*max(ch1), CutOffFreq_indic,'FontSize',10)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('Current * 50\Omega [V]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'OSC 측정 profile, 노이즈 제거버전';' '},'FontSize',15)
            subplot(1,2,2)
            plot(time(1:end-1),net_accum_charge_density, 'k-'), grid on, grid minor
            if ErrorRatio > 0.5*para3 , text(0.6*time(end), 1.1*Q_charge, ErrorRate_indic,'FontSize',12) , end
            text(0.6*time(end), 1*Q_charge, Q_res_indic,'FontSize',12)
            text(0.6*time(end), 0.9*Q_charge, Q_discharge_indic,'FontSize',12)
            text(0.6*time(end), 0.8*Q_charge, Q_charge_indic,'FontSize',12)
            text(0.6*time(end), 0.7*Q_charge, Area_indic,'FontSize',12)
            text(0.6*time(end), 0.6*Q_charge, ErrorRate_indic,'FontSize',12)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('ChargeDensity [C/m^2]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'정전 및 방전용량';' '},'FontSize',15)
            main_title = sgtitle(dat_file, 'Color', 'red','FontSize',20);

        end


        if ErrorRatio > para3 , state = 10; 
        else,  state = -1;   ValueUpdate = 0;  end   % Qres가 기준보다 클 경우 dialog 띄우기

    end


    if state ==2   % Qres 보정본으로 그리기

        Q_charge_indic = strcat('Q_{charge} = ', num2str(NewQ_charge), ' [C/m^2]');
        Q_discharge_indic = strcat('Q_{discharge} = ', num2str(NewQ_discharge), ' [C/m^2]');
        Q_res_indic = strcat('Q_{res} = ', num2str(NewQ_res), ' [C/m^2]');
        Area_indic = strcat('Area_{cap} = ', num2str(cap_area * 1e12), '[\mum^2]');
        ErrorRate_indic = strcat('Qres size = ', num2str(ErrorRatio), '% of Q_{charge}');

        if strcmp(prompt1,'Positive biased') == 1
            fig6 = figure(6);
            fig6.Position = [10 50 0.95*PlotSizeRef(3) 0.8*PlotSizeRef(4)];
            subplot(1,2,1)
            plot(time, ch1, 'k-'), hold on
            plot(time , OSCErrorMean * ones(length(time),1)), hold on
            plot(time, ch1_return_beforeAdj, 'r-', "LineWidth",2), grid on, grid minor , hold off
            text(0.5*time(end), max(ch1), 'FFT info.','FontSize',12)
            text(0.5*time(end), 0.9*max(ch1), MaxFreq_indic,'FontSize',10)
            text(0.5*time(end), 0.8*max(ch1), FreqIncrement_indic,'FontSize',10)
            text(0.5*time(end), 0.7*max(ch1), CutOffFreq_indic,'FontSize',10)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('Current * 50\Omega [V]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'OSC 측정 profile, 노이즈 제거버전';' '},'FontSize',15)
            subplot(1,2,2)
            plot(time(1:end-1), QresAdjust, 'k-'), grid on, grid minor, hold on
            plot(time(Q_charge_position), QresAdjust(Q_charge_position), 'bo'), hold on
            plot(time(Qres_position), QresAdjust(Qres_position), 'bo'), hold off
            text(0.6*time(end), 0.6*Q_charge, ErrorRate_indic,'FontSize',12)
            text(0.6*time(end), 0.7*Q_charge, Q_res_indic,'FontSize',12)
            text(0.6*time(end), 0.8*Q_charge, Q_discharge_indic,'FontSize',12)
            text(0.6*time(end), 0.9*Q_charge, Q_charge_indic,'FontSize',12)
            text(0.6*time(end), Q_charge, Area_indic,'FontSize',12)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('ChargeDensity [C/m^2]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'정전 및 방전용량';' '},'FontSize',15)
            main_title = sgtitle(dat_file, 'Color', 'red','FontSize',20);

        elseif strcmp(prompt1,'Negative biased') == 1
            fig6 = figure(6);
            fig6.Position = [10 50 0.95*PlotSizeRef(3) 0.8*PlotSizeRef(4)];
            subplot(1,2,1)
            plot(time, ch1, 'k-'), hold on
            plot(time , OSCErrorMean * ones(length(time),1)), hold on
            plot(time, ch1_return_beforeAdj, 'r-', "LineWidth",2), grid on, grid minor , hold off
            text(0.1*time(end), max(ch1), 'FFT info.','FontSize',12)
            text(0.1*time(end), 0.9*max(ch1), MaxFreq_indic,'FontSize',10)
            text(0.1*time(end), 0.8*max(ch1), FreqIncrement_indic,'FontSize',10)
            text(0.1*time(end), 0.7*max(ch1), CutOffFreq_indic,'FontSize',10)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('Current * 50\Omega [V]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'OSC 측정 profile, 노이즈 제거버전';' '},'FontSize',15)
            subplot(1,2,2)
            plot(time(1:end-1), QresAdjust, 'k-'), grid on, grid minor
            text(0.6*time(end), 1.1*Q_charge, ErrorRate_indic,'FontSize',12)
            text(0.6*time(end), 1*Q_charge, Q_res_indic,'FontSize',12)
            text(0.6*time(end), 0.9*Q_charge, Q_discharge_indic,'FontSize',12)
            text(0.6*time(end), 0.8*Q_charge, Q_charge_indic,'FontSize',12)
            text(0.6*time(end), 0.7*Q_charge, Area_indic,'FontSize',12)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('ChargeDensity [C/m^2]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'정전 및 방전용량';' '},'FontSize',15)
            main_title = sgtitle(dat_file, 'Color', 'red','FontSize',20);

        end

        RedrawDialog2 = questdlg('진행할까요?', 'Dialog 1',...
            '1. 이대로 저장', '2. 아까껄로' , '2. 아까껄로');
        
        if strcmp(RedrawDialog2(1), '2') == 1 , state = 1; close(fig6) 
        else ,  state = -1;  ValueUpdate = 1;  end

    end



    if state == 10
        
        RedrawDialog = questdlg('큰 Qres가 탐지되었습니다.Qres 보정?', 'Dialog 1',...
            '1. Qres 보정하여 다시 그리기', '2. 그냥 냅둠' , '2. 그냥 냅둠');
        
        if strcmp(RedrawDialog(1), '1') == 1 , state = 2; close(fig6) 
        else ,  state = -1;   ValueUpdate = 0;   end

        
    end


end



dat_file(end-3 : end) = '.jpg';

fname = dat_file_path;    pic_name = dat_file;    
saveas(gcf, fullfile(fname, pic_name), 'jpg');

% ---------------- 데이터 장표 생성 -------------------

file_listing = dir(dat_file_path);
file_listing(1) = [];
file_listing(1) = [];


if ValueUpdate == 1
    Q_charge = NewQ_charge;
    Q_discharge = NewQ_discharge;
    Q_res = NewQ_res;
end


var_exist = 0;
for i = 1:length(file_listing)
    
    if strcmp(file_listing(i).name, 'dataset.mat') == 1
        var_exist = 1;
        break
    end
end

var_state = 0;
while var_state >=0

    if var_exist == 0
        dataset(1).title = dat_file;
        dataset(1).Q_charge = num2str(Q_charge);
        dataset(1).Q_discharge = num2str(Q_discharge);
        dataset(1).Q_res = num2str(Q_res);
        dataset(1).Area = num2str(cap_area * 1e12);

        file_directory = strcat(dat_file_path, 'dataset.mat');

        save(file_directory, "dataset")

        var_state = -1;
    end


    if var_exist == 1
        file_directory = strcat(dat_file_path, 'dataset.mat');

        load(file_directory);

        len = length(dataset);

        dataset(len+1).title = dat_file;
        dataset(len+1).Q_charge = num2str(Q_charge);
        dataset(len+1).Q_discharge = num2str(Q_discharge);
        dataset(len+1).Q_res = num2str(Q_res);
        dataset(len+1).Area = num2str(cap_area * 1e12);

        file_directory = strcat(dat_file_path, 'dataset.mat');

        save(file_directory, "dataset")

        var_state = -1;

    end

end

T = struct2table(dataset);
table_directory = strcat(dat_file_path,'dataset.csv');
writetable(T, table_directory , 'Delimiter', ',')


fclose('all');
clear
close all
clc


%% ===================================== Data Load =======================================
[dat_file,dat_file_path] = uigetfile('*.txt','Select .dat file for TF analyzer data');

file = strcat(dat_file_path, dat_file);
        
fid = fopen(file);


buffer = fgetl(fid);

data = textscan(fid, '%f%f%f', 'Delimiter', '    ');

time = data{1,1};
ch1 = data{1,2};
ch2 = data{1,3};


DefInput = 1/mean(gradient(time));


%% ===================================== condition input and error handle ================================================

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
NCDialogPara.CapacitorArea = num2str(para2);      % [um2]
NCDialogPara.QresTolerance = num2str(para3);          % [%]
NCDialogPara.PulseWidth = num2str(para4);           % [ns]
NCDialogPara.PulseRiseFall = num2str(para5);        % [ns]

save(filePath, "NCDialogPara")


prompt1 = para1;
cap_area = para2 * 1e-12;


%% ===================================== FFT =======================================

y_FFTout = fft(ch1);

Fs = DefInput;
N = length(ch1);              % Get the Number of Points
k = 0:N-1;                  % Create a Vector from 0 to N-1
T = N/Fs;                   % Get the Minimum Frequency
freq = k/T;                 % Create a Vector fot the Frequency Range

PlotSizeRef = get(0,'screensize');

 fig1 = figure(1);
 fig1.Position = [10 50 0.7*PlotSizeRef(3) 0.7*PlotSizeRef(4)];
 subplot(2,1,1)
 plot(freq(2:end), abs(y_FFTout(2:end)))
 subplot(2,1,2)
 semilogx(freq(2:end), abs(y_FFTout(2:end)))


FreqLast = log10(freq(end)/2);     
FreqInit = log10(freq(2));       
FreqSearchRange = linspace(FreqInit, FreqLast, 60)';

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

mse_mean = mean(MSE2);          
[~, idx_closest] = min(abs(MSE2 - mse_mean));


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

figure(4)
fig4 = figure(4);
plot(time, ch1, 'k-'), hold on
plot(time, ch1_return, 'r-', "LineWidth",2), grid on
fig4.Position = [0.3*PlotSizeRef(3) 50 0.5*PlotSizeRef(3) 0.7*PlotSizeRef(4)];



%==================================================================================================================


tap = min(ch1(1:round(0.1*length(time)))) * 1.1;        

ErrorSearchRange = linspace(-tap , tap , 200);

OSCMSE = zeros(length(ErrorSearchRange) , 1);
for j = 1:length(ErrorSearchRange)
    OSCMSE(j) = immse(ch1_return , ErrorSearchRange(j) * ones(length(ch1_return),1)) ;

end


 figure(5)
 plot(ErrorSearchRange, OSCMSE)

OSCErrorMean = ErrorSearchRange(find(OSCMSE == min(OSCMSE)));


 figure(7)
 fig7 = figure(7);
 plot(time, ch1, 'k-'), hold on
 plot(time , OSCErrorMean * ones(length(time),1)), hold on
 plot(time, ch1_return, 'r-', "LineWidth",2), grid on
 fig7.Position = [0.3*PlotSizeRef(3) 50 0.5*PlotSizeRef(3) 0.7*PlotSizeRef(4)];
ch1_return_beforeAdj = ch1_return;
ch1_return = ch1_return - OSCErrorMean * ones(length(time),1);



% if strcmp(prompt1,'Negative biased') == 1
%     ch1_return = -ch1_return;
% end

current_data = ch1_return / 50 ;                    
del_time = time(2:end) - time(1:end-1);
del_charge = del_time .* current_data(1:end-1);    
accum_charge = cumsum(del_charge);                 

net_accum_charge_density = accum_charge / cap_area ;


%% ================================== Find Qchar , Qdis , Qres =======================================


[Q_charge, Q_charge_position] = max(real(net_accum_charge_density));

slopeRange2 = length(time(Q_charge_position: Q_charge_position+ round(0.5*(length(time)-Q_charge_position)))   );
slopeFinder2 = zeros(slopeRange2 , 1);

for i = 1:slopeRange2-1
    slopeFinder2(i) = (net_accum_charge_density(end) - net_accum_charge_density(i+Q_charge_position-1))...
        / (time(end) - time(i+Q_charge_position-1));
end

MaxSlope2 = find(slopeFinder2 == min(slopeFinder2)) + Q_charge_position;

Qres_position = find(time > time(MaxSlope2) + para5*1e-9 + 0.6*para4*1e-9 , 1 , "first");
Q_res = real(min(net_accum_charge_density(Qres_position:end)));
Qres_position = find(net_accum_charge_density == Q_res);
Q_discharge = Q_charge - Q_res ;



%% ========================================= Qres Handle ==========================================

slopeRange = floor(MaxSlope2 * 0.7);
slopeFinder = zeros(slopeRange,1);



for i = 1:slopeRange
    slopeFinder(i) = (net_accum_charge_density(MaxSlope2) - net_accum_charge_density(i)) / (time(MaxSlope2) - time(i)) ;
end

MaxSlope = find(slopeFinder == max(slopeFinder));
MaxSlope = round(0.75*MaxSlope);

polyfit_x = time(1:MaxSlope);           
polyfit_y = net_accum_charge_density(1:MaxSlope);
Linregress = polyfit(polyfit_x, polyfit_y , 1);

QresAdjust = net_accum_charge_density - (time(1:end-1)*Linregress(1) + Linregress(2));
NewQ_charge = QresAdjust(Q_charge_position);
NewQ_res = min(real(QresAdjust(Qres_position:end)));
NewQ_discharge = NewQ_charge - NewQ_res;

ErrorRatio = 100 * NewQ_res / NewQ_charge;



if strcmp(prompt1,'Negative biased') == 1
    ch1_return = -ch1_return;
    net_accum_charge_density = - net_accum_charge_density;
    QresAdjust = -QresAdjust;

    Q_charge = - Q_charge;
    Q_discharge = - Q_discharge; 
    Q_res = - Q_res;

    NewQ_charge = - NewQ_charge;
    NewQ_discharge = - NewQ_discharge;
    NewQ_res = - NewQ_res;
end


%% ========================================= Post processing ================================================

state = 1;
while state > 0

    if state == 1  



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
            title({'OSC measurment profile, noise removed';' '},'FontSize',15)
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
            title({'Accumulation charge density';' '},'FontSize',15)
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
            title({'OSC measurment profile, noise removed';' '},'FontSize',15)
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
            title({'Accumulation charge density';' '},'FontSize',15)
            main_title = sgtitle(dat_file, 'Color', 'red','FontSize',20);

        end


        if ErrorRatio > para3 , state = 10; 
        else,  state = -1;   ValueUpdate = 0;  end

    end


    if state ==2   

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
            title({'OSC measurment profile, noise removed';' '},'FontSize',15)
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
            title({'Accumulation charge density';' '},'FontSize',15)
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
            title({'OSC measurment profile, noise removed';' '},'FontSize',15)
            subplot(1,2,2)
            plot(time(1:end-1), QresAdjust, 'k-'), grid on, grid minor
            text(0.6*time(end), 1.1*Q_charge, ErrorRate_indic,'FontSize',12)
            text(0.6*time(end), 1*Q_charge, Q_res_indic,'FontSize',12)
            text(0.6*time(end), 0.9*Q_charge, Q_discharge_indic,'FontSize',12)
            text(0.6*time(end), 0.8*Q_charge, Q_charge_indic,'FontSize',12)
            text(0.6*time(end), 0.7*Q_charge, Area_indic,'FontSize',12)
            xlabel('Time [s]', 'FontSize', 18, 'FontWeight', 'Bold')
            ylabel('ChargeDensity [C/m^2]', 'FontSize', 18, 'FontWeight', 'Bold')
            title({'Accumulation charge density';' '},'FontSize',15)
            main_title = sgtitle(dat_file, 'Color', 'red','FontSize',20);

        end

        RedrawDialog2 = questdlg('Proceed?', 'Dialog 1',...
            '1. Yes', '2. No' , '2. No');
        
        if strcmp(RedrawDialog2(1), '2') == 1 , state = 1; close(fig6) 
        else ,  state = -1;  ValueUpdate = 1;  end

    end



    if state == 10
        
        RedrawDialog = questdlg('A large Q_res has been detected.\nWould you like to apply Q_res correction?', 'Dialog 1',...
            '1. Re-plot with correction', '2. Keep original' , '2. Keep original');
        
        if strcmp(RedrawDialog(1), '1') == 1 , state = 2; close(fig6) 
        else ,  state = -1;   ValueUpdate = 0;   end

        
    end


end



dat_file(end-3 : end) = '.jpg';

fname = dat_file_path;    pic_name = dat_file;    
saveas(gcf, fullfile(fname, pic_name), 'jpg');


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
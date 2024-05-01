function show_box(wavenumber,match_range,N_coh,N_compressed,N_net,N_RC2,C_coh,C_compressed,C_net,C_RC2,S_coh,S_compressed,S_net,S_RC2)
% 绘制结果的图像
figure;
subplot(3,2,1);
plot(wavenumber, N_coh, 'Color', [0, 0.447, 0.741], 'LineWidth', 1.2); hold on
plot(wavenumber, C_coh, 'Color', [0.851, 0.325, 0.098], 'LineWidth', 1.2); hold on
plot(wavenumber, S_coh, 'Color', [0.467, 0.675, 0.188], 'LineWidth', 1.2); hold on
plot(wavenumber, N_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, C_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, S_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8);
% legend('S1-FFT','S1-CS','S1-Phy','S1-truth')
xlim([1.5,2.3])
ylim([-1.2, 1.2])
% leg = legend('{\itN}','{\itC}','{\itS}','truth');
% leg.ItemTokenSize = [6,5];
% leg.NumColumns = 2;
ylabel('FFT')
xticks([])
set(gca, 'LineWidth', 1, 'Fontsize', 16, 'Fontname', 'Times New Roman', 'Position', [0.12, 0.75, 0.36, 0.23])

subplot(3,2,2);
boxplot([abs(N_coh(match_range)-N_RC2(match_range)) abs(C_coh(match_range)-C_RC2(match_range)) abs(S_coh(match_range)-S_RC2(match_range))],...
    'Symbol','', 'ColorGroup',[0.741,0,0.447]);
ylim([-0.01, 0.052])
xticks([])
set(gca, 'LineWidth', 1, 'Fontsize', 16, 'Fontname', 'Times New Roman', 'Position', [0.66, 0.75, 0.28, 0.23])

subplot(3,2,3);
plot(wavenumber, N_compressed, 'Color', [0, 0.447, 0.741], 'LineWidth', 1.2); hold on
plot(wavenumber, C_compressed, 'Color', [0.851, 0.325, 0.098], 'LineWidth', 1.2); hold on
plot(wavenumber, S_compressed, 'Color', [0.467, 0.675, 0.188], 'LineWidth', 1.2); hold on
plot(wavenumber, N_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, C_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, S_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8);
xlim([1.5,2.3])
ylim([-1.2, 1.2])
% leg = legend('{\itN}','{\itC}','{\itS}','truth');
% leg.ItemTokenSize = [6,5];
% leg.NumColumns = 2;
ylabel('CS')
xticks([])
set(gca, 'LineWidth', 1, 'Fontsize', 16, 'Fontname', 'Times New Roman', 'Position', [0.12, 0.48, 0.36, 0.23])

subplot(3,2,4);
boxplot([abs(N_compressed(match_range)-N_RC2(match_range)) abs(C_compressed(match_range)-C_RC2(match_range)) abs(S_compressed(match_range)-S_RC2(match_range))],...
    'Symbol','', 'ColorGroup',[0.741,0,0.447]);
ylim([-0.01, 0.052])
xticks([])
ylabel('absolute error')
set(gca, 'LineWidth', 1, 'Fontsize', 16, 'Fontname', 'Times New Roman', 'Position', [0.66, 0.48, 0.28, 0.23])

subplot(3,2,5);
plot(wavenumber, N_net, 'Color', [0, 0.447, 0.741], 'LineWidth', 1.2); hold on
plot(wavenumber, C_net, 'Color', [0.851, 0.325, 0.098], 'LineWidth', 1.2); hold on
plot(wavenumber, S_net, 'Color', [0.467, 0.675, 0.188], 'LineWidth', 1.2); hold on
plot(wavenumber, N_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, C_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, S_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8);
xlim([1.5,2.3])
ylim([-1.2, 1.2])
leg = legend('{\itN}','{\itC}','{\itS}','truth');
leg.ItemTokenSize = [6,5];
leg.NumColumns = 4;
ylabel('PCUNN')
% xlabel('wavenumber (\mum^{-1})')
% leg = legend('FFT','CS','Phy','truth','Orientation','horizontal','Location','bestoutside');
% leg.ItemTokenSize = [15,5];
set(gca, 'LineWidth', 1, 'Fontsize', 16, 'Fontname', 'Times New Roman', 'Position', [0.12, 0.21, 0.36, 0.23])

subplot(3,2,6); %'Labels',{'N','C','S'},
boxplot([abs(N_net(match_range)-N_RC2(match_range)) abs(C_net(match_range)-C_RC2(match_range)) abs(S_net(match_range)-S_RC2(match_range))],...
     'Symbol','', 'ColorGroup',[0.741,0,0.447]);
ylim([-0.01, 0.052])
set(gca, 'LineWidth', 1, 'Fontsize', 16, 'Fontname', 'Times New Roman', 'Position', [0.66, 0.21, 0.28, 0.23])
end
function show_initial_final(wavenumber,match_range,N_net_simple,C_net_simple,S_net_simple,N_RC2,C_RC2,S_RC2,N_net,C_net,S_net)
% 绘制net1和net3参数的变化
figure;
subplot(1,2,1)
plot(wavenumber, N_net_simple, 'Color', [0, 0.447, 0.741], 'LineWidth', 1.2); hold on
plot(wavenumber, C_net_simple, 'Color', [0.851, 0.325, 0.098], 'LineWidth', 1.2); hold on
plot(wavenumber, S_net_simple, 'Color', [0.467, 0.675, 0.188], 'LineWidth', 1.2); hold on
plot(wavenumber, N_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, C_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, S_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8);
leg = legend('{\itN}','{\itC}','{\itS}','truth');
leg.ItemTokenSize = [6,5];
xlim([1.5,2.3])
ylim([-1.2, 1.2])
% ylabel('N', 'Rotation', 0)
xlabel('wavenumber (\mum^{-1})')
title('initial {\itN}, {\itC}, {\itS}')
set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
subplot(1,2,2) % 'Labels',{'e(N)','e(C)','e(S)'},
boxplot([abs(N_net_simple(match_range)-N_RC2(match_range)) abs(C_net_simple(match_range)-C_RC2(match_range)) ...
    abs(S_net_simple(match_range)-S_RC2(match_range))],...
     'Symbol','', 'ColorGroup',[0.741,0,0.447]);
ylim([-0.01, 0.052])
title('error of demodulation')
ylabel('absolute error')
set(gca, 'LineWidth', 1, 'Fontsize', 16, 'Fontname', 'Times New Roman')
set(gcf, 'Units', 'centimeters', 'Position', [10 5 16 9])

figure
subplot(1,2,1)
plot(wavenumber, N_net, 'Color', [0, 0.447, 0.741], 'LineWidth', 1.2); hold on
plot(wavenumber, C_net, 'Color', [0.851, 0.325, 0.098], 'LineWidth', 1.2); hold on
plot(wavenumber, S_net, 'Color', [0.467, 0.675, 0.188], 'LineWidth', 1.2); hold on
plot(wavenumber, N_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, C_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8); hold on
plot(wavenumber, S_RC2, '--', 'Color', [0, 0, 0], 'LineWidth', 0.8);
% legend('N-net','C-net','S-net','N-truth','C-truth','S-truth')
leg = legend('{\itN}','{\itC}','{\itS}','truth');
leg.ItemTokenSize = [6,5];
xlim([1.5,2.3])
ylim([-1.2, 1.2])
% ylabel('N', 'Rotation', 0)
xlabel('wavenumber (\mum^{-1})')
title('final {\itN}, {\itC}, {\itS}')
set(gca, 'Fontsize', 16, 'Fontname', 'Times New Roman')
subplot(1,2,2) % 'Labels',{'e(N)','e(C)','e(S)'},
boxplot([abs(N_net(match_range)-N_RC2(match_range)) abs(C_net(match_range)-C_RC2(match_range)) ...
    abs(S_net(match_range)-S_RC2(match_range))],...
     'Symbol','', 'ColorGroup',[0.741,0,0.447]);
ylim([-0.01, 0.052])
ylabel('absolute error')
title('error of demodulation')
set(gcf, 'Units', 'centimeters', 'Position', [10 5 16 9])
set(gca, 'LineWidth', 1, 'Fontsize', 16, 'Fontname', 'Times New Roman')
end
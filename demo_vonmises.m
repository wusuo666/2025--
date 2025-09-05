% 温·米塞斯（von Mises）分布示意图
% 展示不同浓度参数 kappa 下的概率密度曲线与极坐标形状
% PDF(theta; mu, kappa) = exp(kappa*cos(theta-mu)) / (2*pi*I0(kappa))
% 其中 I0 为第一类零阶修正贝塞尔函数（besseli(0, x)）

clear; clc;

% 参数设置
mu = pi/2;                         % 均值方向（弧度）
kappas = [0, 0.5, 2,4,6 8];           % 浓度参数（越大越集中，kappa=0 为均匀分布）

% 离散角度
theta = linspace(-pi, pi, 2000);

% 计算各 kappa 的 PDF
pdfs = zeros(length(kappas), numel(theta));
for i = 1:length(kappas)
    k = kappas(i);
    c = 1./(2*pi*besseli(0, k));           % 归一化常数
    pdfs(i, :) = c .* exp(k .* cos(theta - mu));
end

% 准备布局：左侧笛卡尔坐标，右侧极坐标
fig = figure('Name','von Mises PDF','Position',[100,100,900,360]);
t = tiledlayout(fig, 1, 2, 'TileSpacing','compact','Padding','compact');

% 左：不同 kappa 的 PDF 曲线
ax1 = nexttile(t, 1);
hold(ax1, 'on'); grid(ax1, 'on');
cols = lines(length(kappas));
for i = 1:length(kappas)
    plot(ax1, theta, pdfs(i,:), 'LineWidth', 1.8, 'Color', cols(i,:));
end
xlabel(ax1, '\theta (rad)'); ylabel(ax1, 'PDF(\theta)');
% 注意：sprintf中使用 \\mu 以便TeX解释器显示希腊字母
title(ax1, sprintf('von Mises PDF  (\\mu=%.2f rad)', mu), 'Interpreter','tex');
labels = arrayfun(@(k) sprintf('%s=%.1f', char(954), k), kappas, 'UniformOutput', false); % 954=κ
legend(ax1, labels, 'Location','northwest');
xlim(ax1, [-pi, pi]);
set(ax1,'XTick',-pi:pi/2:pi,'XTickLabel',{'-\pi','-\pi/2','0','\pi/2','\pi'});

% 右：极坐标下的 PDF（半径=PDF）
ax2 = polaraxes(t);
ax2.Layout.Tile = 2;
hold(ax2, 'on');
for i = 1:length(kappas)
    polarplot(ax2, theta, pdfs(i,:), 'LineWidth', 1.8, 'Color', cols(i,:));
end
rlim(ax2, [0, max(pdfs(:))*1.05]);
title(ax2, '极坐标表示（半径=PDF）');
legend(ax2, labels, 'Location','southoutside');

% 显示均值方向
rmax = max(pdfs(:))*1.05;
polarplot(ax2, [mu mu], [0, rmax], 'k--', 'LineWidth', 1.2);

% 提示
fprintf('已生成 von Mises 分布的 PDF 曲线与极坐标示意图（mu=%.2f）。\n', mu);
fprintf('你可以修改 mu 与 kappas 观察均值方向与集中度变化的影响。\n'); 
%% Question 1: 计算遮挡时长并可视化遮挡区间
% 本脚本根据投放瞬间的初始位置与速度，计算遮挡时长并可视化遮挡过程

% 投放瞬间的初始位置/速度（示例，根据实际需求调整）
init = struct();
init.observerPos    = pos_M1_bao;       % 导弹M1雷达位置
init.observerVel    = vv_M1;            % 导弹M1雷达速度
init.cylinderCenter = [0, 200, h_target/2]; % 目标圆柱体底面中心
init.cylinderVel    = [0, 0, 0];        % 目标圆柱体速度（静止）
init.cylinderRadius = r_target;         % 目标圆柱体半径
init.cylinderHeight = h_target;         % 目标圆柱体高度
init.cylinderDir    = [0, 0, 1];        % 目标圆柱体方向（垂直向上）
init.sphereCenter   = pos_bao;          % 诱饵云团中心
init.sphereVel      = vv_bao;           % 诱饵云团速度
init.sphereRadius   = r_cloud;          % 诱饵云团半径

% 计时开始
tic();

% 计算遮挡时长和区间（使用固定步长方法）
t0 = 0;
t1 = 10;  % 总仿真时间10秒
dt = 0.01; % 计算步长0.01秒
[totalDuration, intervals] = computeOcclusionFixedStep(t0, t1, dt, init);

% 计时结束
elapsed = toc();
fprintf('用时 %.6f 秒。\n', elapsed);

% 输出结果
fprintf('遮挡总时长: %.6f s\n', totalDuration);
if isempty(intervals)
    fprintf('无遮挡\n');
else
    disp('遮挡区间 [t_begin, t_end] (s):');
    disp(intervals);
end

%% 可视化部分

% 1. 绘制遮挡区间时间线
figure('Name', '遮挡区间可视化', 'Position', [100, 500, 800, 300]);

% 绘制时间轴
subplot(2,1,1);
hold on;
plot([t0, t1], [0, 0], 'k-', 'LineWidth', 1.5); % 时间轴
for i = 1:size(intervals, 1)
    plot(intervals(i,:), [0, 0], 'r-', 'LineWidth', 5); % 遮挡区间
    text(intervals(i,1), 0.1, sprintf('%.3f', intervals(i,1)), 'FontSize', 9);
    text(intervals(i,2), 0.1, sprintf('%.3f', intervals(i,2)), 'FontSize', 9);
end
title(sprintf('遮挡区间 (总时长: %.6f s)', totalDuration));
xlim([t0, t1]);
ylim([-0.5, 0.5]);
set(gca, 'YTick', []);
xlabel('时间 (s)');
grid on;

% 2. 三维场景可视化（关键时刻）
if ~isempty(intervals)
    % 选择关键时刻进行可视化
    key_times = [];
    for i = 1:size(intervals, 1)
        % 为每个区间添加起点、中点和终点
        key_times = [key_times, intervals(i,1), mean(intervals(i,:)), intervals(i,2)];
    end
    
    % 添加一个非遮挡时刻用于对比
    if key_times(1) > t0 + 0.1
        key_times = [t0, key_times];
    elseif key_times(end) < t1 - 0.1
        key_times = [key_times, t1];
    end
    
    % 只保留最多5个关键时刻
    if length(key_times) > 5
        % 保留每个区间的开始和结束
        reduced_times = [];
        for i = 1:size(intervals, 1)
            reduced_times = [reduced_times, intervals(i,1), intervals(i,2)];
        end
        key_times = reduced_times;
    end
    
    % 可视化关键时刻
    subplot(2,1,2);
    visualizeTimePoints(init, key_times);
else
    subplot(2,1,2);
    text(0.5, 0.5, '无遮挡区间', 'FontSize', 14, 'HorizontalAlignment', 'center');
    axis off;
end

%% 创建动画（可选，取消注释运行）
% 创建遮挡过程动画
% anim_times = t0:0.1:t1;  % 以0.1秒为步长采样
% figure('Name', '遮挡过程动画');
% visualizeOcclusion(init, anim_times);

function visualizeTimePoints(init, times)
    hold on;
    colors = jet(length(times)); % 使用彩色区分不同时刻
    
    for i = 1:length(times)
        t = times(i);
        
        % 计算当前时刻位置
        obsPos = init.observerPos + t * init.observerVel;
        cylC = init.cylinderCenter + t * init.cylinderVel;
        sphC = init.sphereCenter + t * init.sphereVel;
        
        % 检查是否遮挡
        isBlocked = isCylinderBlockedBySphere( ...
            obsPos, cylC, init.cylinderRadius, init.cylinderHeight, init.cylinderDir, ...
            sphC, init.sphereRadius);
        
        % 绘制观察者到圆柱体中心的线
        line([obsPos(1), cylC(1)], [obsPos(2), cylC(2)], [obsPos(3), cylC(3)], ...
             'Color', colors(i,:), 'LineWidth', 1, 'LineStyle', ':');
        
        % 标注时间点
        if isBlocked
            status = '(遮挡)';
        else
            status = '(未遮挡)';
        end
        text(obsPos(1), obsPos(2), obsPos(3) + 2, ...
             sprintf('t=%.2fs %s', t, status), ...
             'Color', colors(i,:), 'FontSize', 8);
    end
    
    % 添加图例和标题
    title('关键时刻的相对位置');
    grid on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    view(30, 30); % 设置视角
    axis equal;
end 
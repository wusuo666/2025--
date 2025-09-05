%% 遮挡过程可视化示例
% 本脚本展示如何使用visualizeOcclusion函数创建静态图与动画

%% 场景1: 静态状态 - 展示指定时刻的遮挡情况
% 设置场景
init = struct();
init.observerPos = [0, 0, 0];           % 观察者位置
init.observerVel = [0, 0, 0];           % 观察者速度（静止）

init.cylinderCenter = [5, 0, 0];        % 圆柱体底面中心
init.cylinderVel = [0, 0, 0];           % 圆柱体速度（静止）
init.cylinderRadius = 1.0;              % 圆柱体半径
init.cylinderHeight = 3.0;              % 圆柱体高度
init.cylinderDir = [0, 0, 1];           % 圆柱体方向（沿z轴）

init.sphereCenter = [3, 0, 0];          % 球体中心
init.sphereVel = [0, 0, 0];             % 球体速度（静止）
init.sphereRadius = 1.5;                % 球体半径

% 创建静态图（t=0时刻）
visualizeOcclusion(init, 0);

%% 场景2: 移动球体导致遮挡状态变化的动画
init2 = init;
init2.sphereCenter = [8, 0, 0];         % 球体初始位置在圆柱体右侧
init2.sphereVel = [-0.5, 0, 0];         % 球体向左移动

% 创建动画 - 缩短时间范围，减少帧数提高效率
times = 0:1:10;  % 从0到10秒，每1秒一帧
% 可视化但不保存动画
visualizeOcclusion(init2, times);

% 如需保存动画，可以使用：
% visualizeOcclusion(init2, times, 'SaveAnimation', 'sphere_animation');

%% 场景3: 观察者环绕场景的动画
init3 = struct();
init3.observerPos = [10, 0, 0];         % 观察者初始位置
init3.observerVel = [0, 0, 0];          % 不使用速度，将手动更新位置

init3.cylinderCenter = [0, 0, 0];       % 圆柱体在原点
init3.cylinderVel = [0, 0, 0];
init3.cylinderRadius = 1.0;
init3.cylinderHeight = 3.0;
init3.cylinderDir = [0, 0, 1];

init3.sphereCenter = [-3, 0, 0];        % 球体在圆柱体左侧
init3.sphereVel = [0, 0, 0];
init3.sphereRadius = 1.5;

% 创建环绕动画（观察者绕y轴旋转）
nFrames = 30; % 减少帧数提高效率
times = 1:nFrames;
fig = figure('Position', [100, 100, 800, 600]);

% 预先计算观察者位置
radius = 10;
angleStep = 2*pi/nFrames;

% 不保存所有帧，只显示动画
for i = 1:nFrames
    angle = (i-1) * angleStep;
    
    % 更新观察者位置（绕圆圈运动）
    init3.observerPos = [radius * cos(angle), radius * sin(angle), 0];
    
    % 绘制当前帧
    if i > 1
        clf(fig);
    end
    
    % 使用子图1绘制俯视图
    subplot(1, 2, 1);
    visualizeTopView(init3, i);
    
    % 使用子图2绘制3D视图
    subplot(1, 2, 2);
    visualizeOcclusion(init3, i);
    
    % 添加整体标题
    sgtitle(sprintf('观察者环绕场景 - 帧 %d/%d', i, nFrames), 'FontSize', 16);
    
    % 暂停以控制动画速度
    pause(0.1);
    drawnow;
end

%% 辅助函数：绘制俯视图（2D）
function visualizeTopView(init, t)
    % 计算当前位置
    obsPos = init.observerPos;
    cylPos = init.cylinderCenter;
    sphPos = init.sphereCenter;
    
    % 创建2D俯视图
    plot(obsPos(1), obsPos(2), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    hold on;
    
    % 绘制圆柱体（圆）
    theta = linspace(0, 2*pi, 100);
    x = cylPos(1) + init.cylinderRadius * cos(theta);
    y = cylPos(2) + init.cylinderRadius * sin(theta);
    fill(x, y, 'g', 'FaceAlpha', 0.3);
    
    % 绘制球体（圆）
    x = sphPos(1) + init.sphereRadius * cos(theta);
    y = sphPos(2) + init.sphereRadius * sin(theta);
    fill(x, y, 'r', 'FaceAlpha', 0.3);
    
    % 从观察者到圆柱体边缘的视线
    for angle = 0:15:345
        cylEdgeX = cylPos(1) + init.cylinderRadius * cosd(angle);
        cylEdgeY = cylPos(2) + init.cylinderRadius * sind(angle);
        line([obsPos(1), cylEdgeX], [obsPos(2), cylEdgeY], 'Color', [0.5, 0.5, 0.5], 'LineStyle', '--');
    end
    
    % 绘制圆柱体到球体的连线
    line([cylPos(1), sphPos(1)], [cylPos(2), sphPos(2)], 'Color', 'k', 'LineStyle', ':');
    
    % 检查是否遮挡并添加标题
    isBlocked = isCylinderBlockedBySphere(...
        obsPos, cylPos, init.cylinderRadius, init.cylinderHeight, init.cylinderDir, ...
        sphPos, init.sphereRadius);
    
    if isBlocked
        title('俯视图（圆柱体被遮挡）', 'FontSize', 12);
    else
        title('俯视图（圆柱体未完全被遮挡）', 'FontSize', 12);
    end
    
    % 设置坐标轴
    axis equal;
    grid on;
    xlabel('X');
    ylabel('Y');
    legend('观察者', '圆柱体', '球体');
    
    % 设置合理的轴范围
    maxR = max([norm(obsPos(1:2)), norm(cylPos(1:2))+init.cylinderRadius, norm(sphPos(1:2))+init.sphereRadius]);
    xlim([-maxR*1.2, maxR*1.2]);
    ylim([-maxR*1.2, maxR*1.2]);
    
    hold off;
end 
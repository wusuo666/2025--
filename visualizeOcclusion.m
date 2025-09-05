function fig = visualizeOcclusion(init, tValues, varargin)
% 可视化遮挡过程，展示观察者、球体、圆柱体，并勾勒视角光锥与遮挡
%
% 输入：
%   init     : 结构体，包含初始条件
%              .observerPos, .observerVel
%              .cylinderCenter, .cylinderVel, .cylinderRadius, .cylinderHeight, .cylinderDir
%              .sphereCenter, .sphereVel, .sphereRadius
%   tValues  : 标量或向量，指定要可视化的时刻（秒）
%              - 标量: 静态图，显示单一时刻的遮挡状态
%              - 向量: 动画，显示多个时刻的遮挡状态
%
% 可选名值对参数：
%   'LightCone'      : 逻辑值，是否显示视角光锥，默认true
%   'ConeDetail'     : 光锥采样角度数，默认24
%   'Animation'      : 逻辑值，启用动画效果，默认为tValues长度>1
%   'AxisLimits'     : 1×6 向量 [xmin xmax ymin ymax zmin zmax]，坐标轴范围，默认自动
%   'ShowBlocked'    : 逻辑值，是否显示"遮挡/未遮挡"状态文本，默认true
%   'SaveAnimation'  : 字符串，保存动画的文件名（不带扩展名），默认不保存
%   'FrameRate'      : 动画帧率，默认10
%   'Title'          : 字符串，图表标题，默认自动生成

    % 默认参数
    p = inputParser;
    p.addParameter('LightCone', true, @islogical);
    p.addParameter('ConeDetail', 24, @isnumeric);
    p.addParameter('Animation', numel(tValues) > 1, @islogical);
    p.addParameter('AxisLimits', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 6));
    p.addParameter('ShowBlocked', true, @islogical);
    p.addParameter('SaveAnimation', '', @ischar);
    p.addParameter('FrameRate', 10, @isnumeric);
    p.addParameter('Title', '', @ischar);
    p.parse(varargin{:});
    opts = p.Results;
    
    % 创建图形窗口
    fig = figure('Color', 'w', 'Position', [100, 100, 800, 600]);
    ax = axes('NextPlot', 'add', 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on');
    
    % 为动画准备视频对象
    videoObj = [];
    try
        if opts.Animation && ~isempty(opts.SaveAnimation)
            % 使用更兼容的AVI格式代替MP4，Motion JPEG更加兼容
            videoFile = [opts.SaveAnimation '.avi'];
            videoObj = VideoWriter(videoFile, 'Motion JPEG AVI');
            videoObj.FrameRate = opts.FrameRate;
            videoObj.Quality = 95;
            open(videoObj);
            fprintf('准备录制动画到: %s\n', videoFile);
        end
    catch vidErr
        warning('初始化视频对象失败: %s', vidErr.message);
        videoObj = [];
    end
    
    % 主循环：每个时间点处理一帧
    for t_idx = 1:length(tValues)
        t = tValues(t_idx);
        
        % 清除前一帧（仅动画模式）
        if opts.Animation && t_idx > 1
            cla(ax);
        end
        
        % 计算当前时刻的状态
        obsPos = init.observerPos + t * init.observerVel;
        cylC = init.cylinderCenter + t * init.cylinderVel;
        cylTop = cylC + init.cylinderHeight * init.cylinderDir;
        sphC = init.sphereCenter + t * init.sphereVel;
        
        % 检查是否遮挡
        isBlocked = isCylinderBlockedBySphere( ...
            obsPos, cylC, init.cylinderRadius, init.cylinderHeight, init.cylinderDir, ...
            sphC, init.sphereRadius);
        
        % 绘制观察者（蓝色点）
        plot3(obsPos(1), obsPos(2), obsPos(3), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
        
        % 绘制球体（半透明红色）
        [X, Y, Z] = sphere(30);
        surf(X * init.sphereRadius + sphC(1), ...
             Y * init.sphereRadius + sphC(2), ...
             Z * init.sphereRadius + sphC(3), ...
             'FaceColor', 'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none');
        
        % 绘制圆柱体（半透明绿色）
        [X, Y, Z] = cylinder(init.cylinderRadius, 30);
        % 调整高度并根据方向向量旋转
        Z = Z * init.cylinderHeight;
        
        % 计算旋转矩阵（将z轴[0,0,1]旋转到cylinderDir）
        cylDir = init.cylinderDir / norm(init.cylinderDir);
        defaultDir = [0, 0, 1];
        
        % 如果方向不是默认z轴，计算旋转
        if norm(cylDir - defaultDir) > 1e-10
            rotAxis = cross(defaultDir, cylDir);
            rotAxis = rotAxis / norm(rotAxis);
            rotAngle = acos(dot(defaultDir, cylDir));
            
            % 使用罗德里格斯公式旋转点
            for i = 1:size(X, 1)
                for j = 1:size(X, 2)
                    point = [X(i,j), Y(i,j), Z(i,j)];
                    % 罗德里格斯旋转公式
                    rotatedPoint = point * cos(rotAngle) + ...
                                   cross(rotAxis, point) * sin(rotAngle) + ...
                                   rotAxis * dot(rotAxis, point) * (1 - cos(rotAngle));
                    X(i,j) = rotatedPoint(1);
                    Y(i,j) = rotatedPoint(2);
                    Z(i,j) = rotatedPoint(3);
                end
            end
        end
        
        % 平移到圆柱体位置
        X = X + cylC(1);
        Y = Y + cylC(2);
        Z = Z + cylC(3);
        
        % 绘制圆柱体
        surf(X, Y, Z, 'FaceColor', 'g', 'FaceAlpha', 0.4, 'EdgeColor', 'none');
        
        % 计算并绘制光锥（从观察者到圆柱体）
        if opts.LightCone
            % 找到圆柱体轴垂直的两个方向向量
            [u, v] = generateOrthogonalVectors(init.cylinderDir);
            
            % 底面和顶面的中心
            bottomCenter = cylC;
            topCenter = cylTop;
            
            % 绘制底面和顶面圆的边缘
            theta = linspace(0, 2*pi, opts.ConeDetail);
            bottom = zeros(opts.ConeDetail, 3);
            top = zeros(opts.ConeDetail, 3);
            
            for i = 1:opts.ConeDetail
                bottom(i,:) = bottomCenter + init.cylinderRadius * (cos(theta(i)) * u + sin(theta(i)) * v);
                top(i,:) = topCenter + init.cylinderRadius * (cos(theta(i)) * u + sin(theta(i)) * v);
                
                % 绘制从观察者到边缘点的视线
                line([obsPos(1), bottom(i,1)], [obsPos(2), bottom(i,2)], [obsPos(3), bottom(i,3)], ...
                     'Color', getLineColor(isBlocked, obsPos, bottom(i,:), sphC, init.sphereRadius), ...
                     'LineStyle', '-', 'LineWidth', 0.5);
                
                line([obsPos(1), top(i,1)], [obsPos(2), top(i,2)], [obsPos(3), top(i,3)], ...
                     'Color', getLineColor(isBlocked, obsPos, top(i,:), sphC, init.sphereRadius), ...
                     'LineStyle', '-', 'LineWidth', 0.5);
            end
            
            % 绘制圆柱体边缘轮廓
            plot3(bottom(:,1), bottom(:,2), bottom(:,3), 'g-', 'LineWidth', 2);
            plot3(top(:,1), top(:,2), top(:,3), 'g-', 'LineWidth', 2);
            
            % 绘制连接底面和顶面的竖线（侧棱）
            for i = 1:5:opts.ConeDetail
                line([bottom(i,1), top(i,1)], [bottom(i,2), top(i,2)], [bottom(i,3), top(i,3)], ...
                     'Color', 'g', 'LineWidth', 1.5);
            end
        end
        
        % 显示状态文本
        if opts.ShowBlocked
            if isBlocked
                title(sprintf('时刻 t = %.2f s: 圆柱体完全被遮挡', t), 'FontSize', 14);
            else
                title(sprintf('时刻 t = %.2f s: 圆柱体未完全被遮挡', t), 'FontSize', 14);
            end
        elseif ~isempty(opts.Title)
            title(opts.Title, 'FontSize', 14);
        end
        
        % 添加图例和标签
        xlabel('X', 'FontSize', 12);
        ylabel('Y', 'FontSize', 12);
        zlabel('Z', 'FontSize', 12);
        legend('观察者', '球体', '圆柱体', 'FontSize', 12);
        
        % 设置视角和轴范围
        view(30, 20);
        
        if ~isempty(opts.AxisLimits)
            axis(opts.AxisLimits);
        else
            axis equal;
            % 自动确定合理的轴范围（包含所有物体和考虑运动）
            allPoints = [
                obsPos;
                cylC; cylTop;
                sphC;
            ];
            
            % 考虑球体半径和圆柱体半径
            maxRadius = max(init.sphereRadius, init.cylinderRadius);
            
            % 扩展轴范围
            axisMin = min(allPoints) - maxRadius - 2;
            axisMax = max(allPoints) + maxRadius + 2;
            axis([axisMin(1), axisMax(1), axisMin(2), axisMax(2), axisMin(3), axisMax(3)]);
        end
        
        % 强制渲染
        drawnow;
        
        % 保存当前帧到视频（如果启用）
        try
            if ~isempty(videoObj)
                % 更可靠的帧捕获方式
                frame = getframe(fig);
                % 检查帧是否有效
                if ~isempty(frame.cdata)
                    writeVideo(videoObj, frame);
                    if t_idx == 1 || mod(t_idx, 10) == 0
                        fprintf('已保存帧 %d/%d\n', t_idx, length(tValues));
                    end
                else
                    warning('捕获的帧为空，跳过');
                end
            end
        catch frameErr
            warning('保存帧 %d 失败: %s', t_idx, frameErr.message);
        end
    end
    
    % 关闭视频文件
    if ~isempty(videoObj)
        try
            close(videoObj);
            fprintf('动画已成功保存到: %s\n', [opts.SaveAnimation '.avi']);
        catch closeErr
            warning('关闭视频文件失败: %s', closeErr.message);
        end
    end
end

function color = getLineColor(isFullyBlocked, observer, point, sphereCenter, sphereRadius)
    % 根据视线是否被遮挡返回颜色
    % 绿色: 未遮挡
    % 红色: 遮挡
    % 黄色: 全局遮挡但局部视线未遮挡
    
    % 检查从观察者到特定点的视线是否被球体遮挡
    lineBlocked = isPointBlockedBySphere(observer, point, sphereCenter, sphereRadius);
    
    if isFullyBlocked
        if lineBlocked
            color = [0.8, 0, 0]; % 红色（全局遮挡，该线也被遮挡）
        else
            color = [0.8, 0.8, 0]; % 黄色（全局遮挡，但该线未被遮挡）
        end
    else
        if lineBlocked
            color = [0.5, 0, 0.5]; % 紫色（未全局遮挡，但该线被遮挡）
        else
            color = [0, 0.7, 0]; % 绿色（未遮挡）
        end
    end
end

function [u, v] = generateOrthogonalVectors(w)
    % 生成与给定向量w正交的两个单位向量
    
    % 确保w是单位向量
    w = w / norm(w);
    
    % 找到非平行于w的向量
    if abs(w(1)) < abs(w(2)) && abs(w(1)) < abs(w(3))
        v_temp = [1, 0, 0];
    elseif abs(w(2)) < abs(w(3))
        v_temp = [0, 1, 0];
    else
        v_temp = [0, 0, 1];
    end
    
    % 计算第一个正交向量
    u = cross(w, v_temp);
    u = u / norm(u);
    
    % 计算第二个正交向量
    v = cross(w, u);
    v = v / norm(v);
end

function blocked = isPointBlockedBySphere(observer, point, sphereCenter, sphereRadius)
    % 判断从观察者到指定点的视线是否被球体遮挡
    
    % 计算观察者到点的方向向量
    direction = point - observer;
    distance = norm(direction);
    direction = direction / distance;
    
    % 线段与球体相交的解析判定
    a = 1; % dot(direction, direction) = 1，因为direction是单位向量
    b = 2 * dot(direction, observer - sphereCenter);
    c = dot(observer - sphereCenter, observer - sphereCenter) - sphereRadius^2;
    
    discriminant = b^2 - 4 * a * c;
    
    if discriminant < 0
        % 无交点
        blocked = false;
        return;
    end
    
    % 计算交点
    t1 = (-b - sqrt(discriminant)) / (2 * a);
    t2 = (-b + sqrt(discriminant)) / (2 * a);
    
    % 只要有交点落在[0, distance]区间内，则该点被遮挡
    blocked = (t1 >= 0 && t1 <= distance) || (t2 >= 0 && t2 <= distance && t1 < 0);
end 
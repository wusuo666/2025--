function isCompletelyBlocked = isCylinderBlockedBySphere(observerPos, cylinderCenter, cylinderRadius, cylinderHeight, cylinderDirection, sphereCenter, sphereRadius)
    % 判断圆柱体是否被球体完全遮挡
    % 输入参数：
    % observerPos: 观察点坐标 [x, y, z]
    % cylinderCenter: 圆柱体底面中心坐标 [x, y, z]
    % cylinderRadius: 圆柱体半径
    % cylinderHeight: 圆柱体高度
    % cylinderDirection: 圆柱体轴向单位向量 [x, y, z]
    % sphereCenter: 球体中心坐标 [x, y, z]
    % sphereRadius: 球体半径
    % 输出参数：
    % isCompletelyBlocked: 如果圆柱体完全被球体遮挡则为true，否则为false
    
    % 归一化圆柱体方向向量
    cylinderDirection = cylinderDirection / norm(cylinderDirection);
    
    % 计算圆柱体两个底面的中心点
    bottomCenter = cylinderCenter;
    topCenter = cylinderCenter + cylinderHeight * cylinderDirection;
    
    % 找到圆柱体轴垂直的两个方向向量
    [u, v] = generateOrthogonalVectors(cylinderDirection);
    
    % ------------------- 圆周非均匀（偏置）采样策略 -------------------
    % 思路：在从圆心指向观察者方向的两侧（左右侧，即相差±pi/2）加密采样，
    % 这些位置最有可能形成轮廓并“探出”遮挡范围
    numSamplesCircle = 16;         % 总体采样点数（可调）
    biasFraction = 0.6;            % 分配给两侧加密区的比例（可调，0~1）
    windowWidth = pi/3;            % 每个加密窗口的角宽度（可调）
    % 基于底面圆心计算观察方向在(u,v)平面内的方位角
    c2o = observerPos - bottomCenter;
    c2o_uv = [dot(c2o, u), dot(c2o, v)];
    if norm(c2o_uv) < 1e-12
        % 观察者恰好在轴线方向上，退化情况：给一个固定角度
        theta0 = 0;
    else
        theta0 = atan2(c2o_uv(2), c2o_uv(1));
    end
    angles = generateBiasedCircleAngles(numSamplesCircle, theta0, biasFraction, windowWidth);
    
    % ------------------- 检查底面圆的边缘点 -------------------
    for k = 1:numel(angles)
        angle = angles(k);
        point = bottomCenter + cylinderRadius * (cos(angle) * u + sin(angle) * v);
        if ~isPointBlockedBySphere(observerPos, point, sphereCenter, sphereRadius)
            isCompletelyBlocked = false;
            return;
        end
    end
    
    % ------------------- 检查顶面圆的边缘点 -------------------
    for k = 1:numel(angles)
        angle = angles(k);
        point = topCenter + cylinderRadius * (cos(angle) * u + sin(angle) * v);
        if ~isPointBlockedBySphere(observerPos, point, sphereCenter, sphereRadius)
            isCompletelyBlocked = false;
            return;
        end
    end
    
    % ------------------- 侧棱仅判断上下两点 -------------------
    % 定义与观察者相关的三个“侧棱”方向：
    % 1) 最靠近观察者的侧方向 +observerToCylinder
    % 2) 最远离观察者的侧方向 -observerToCylinder
    % 3) 垂直于视线（在圆周上形成左右两侧轮廓）的方向 ±perpVector
    observerToCylinder = bottomCenter - observerPos;
    observerToCylinder = observerToCylinder - dot(observerToCylinder, cylinderDirection) * cylinderDirection;
    if norm(observerToCylinder) > 1e-10
        observerToCylinder = observerToCylinder / norm(observerToCylinder);
        perpVector = cross(cylinderDirection, observerToCylinder);
        if norm(perpVector) > 1e-10
            perpVector = perpVector / norm(perpVector);
        else
            perpVector = [0, 0, 0];
        end
        
        % 需要检查的侧向单位向量集合
        sideDirs = [ observerToCylinder; -observerToCylinder; perpVector; -perpVector ];
        
        % 仅在每条“侧棱”的上下端点采样（h=0 和 h=1）
        hs = [0, 1];
        for s = 1:size(sideDirs, 1)
            dir = sideDirs(s, :);
            if norm(dir) < 1e-12
                continue;
            end
            for h = hs
                point = bottomCenter + h * cylinderHeight * cylinderDirection + cylinderRadius * dir;
                if ~isPointBlockedBySphere(observerPos, point, sphereCenter, sphereRadius)
                    isCompletelyBlocked = false;
                    return;
                end
            end
        end
    end
    
    % 如果所有检查点都被遮挡，那么圆柱体被球体完全遮挡
    isCompletelyBlocked = true;
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

function angles = generateBiasedCircleAngles(numSamples, theta0, biasFraction, windowWidth)
    % 生成非均匀（偏置）圆周采样角度
    % theta0：从圆心指向观察者在(u,v)平面的方位角
    % 左右两侧中心分别为 theta0 ± pi/2
    
    % 基本参数规范
    numSamples = max(8, round(numSamples));                 % 最少保证8个点
    biasFraction = min(max(biasFraction, 0), 0.95);         % 限制在[0, 0.95]
    windowWidth = min(max(windowWidth, pi/12), pi);         % 限制窗口宽度
    
    % 分配样本数
    n_bias_each = max(2, round(numSamples * biasFraction / 2));
    n_bias_total = 2 * n_bias_each;
    n_uniform = max(0, numSamples - n_bias_total);
    
    % 两个加密中心
    theta1 = wrapTo2Pi(theta0 + pi/2);
    theta2 = wrapTo2Pi(theta0 - pi/2);
    
    % 加密窗口角度
    win1 = linspace(theta1 - windowWidth/2, theta1 + windowWidth/2, n_bias_each);
    win2 = linspace(theta2 - windowWidth/2, theta2 + windowWidth/2, n_bias_each);
    
    % 均匀角度
    if n_uniform > 0
        uni = linspace(0, 2*pi, n_uniform + 1); % 尾首重复点去掉
        uni(end) = [];
    else
        uni = [];
    end
    
    % 合并并归一化到[0, 2*pi)
    angles = [win1, win2, uni];
    angles = arrayfun(@wrapTo2Pi, angles);
    
    % 为稳定性进行排序（非必要）
    angles = sort(angles);
end

function a = wrapTo2Pi(a)
    % 将角度归一化到[0, 2*pi)
    a = mod(a, 2*pi);
    if a < 0
        a = a + 2*pi;
    end
end 
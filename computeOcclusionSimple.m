function [totalDuration, intervals] = computeOcclusionSimple(t0, t1, init)
% 简化版：基于匀速模型，自适应步长搜索遮挡区间，并返回总时长
% 直接调用 isCylinderBlockedBySphere 作为遮挡判定器
%
% 输入：
%   t0, t1    : 起止时间（秒）
%   init      : 结构体，给出投放瞬间的初始位置和速度：
%               .observerPos(1x3), .observerVel(1x3)
%               .cylinderCenter(1x3), .cylinderVel(1x3)
%               .cylinderRadius, .cylinderHeight, .cylinderDir(1x3)
%               .sphereCenter(1x3), .sphereVel(1x3)
%               .sphereRadius
%
% 输出：
%   totalDuration : 遮挡总时长（秒）
%   intervals     : n×2 遮挡时间区间 [tbeg, tend]

    % --------- 自适应步长参数（简单可调） ---------
    dtInit = max((t1 - t0)/200, eps);   % 初始步长
    dtMin  = max((t1 - t0)/1e6, eps);   % 最小步长
    grow   = 1.8;                       % 放大因子
    shrink = 0.25;                      % 缩小因子
    tolRef = max((t1 - t0)/1e7, eps);   % 二分细化阈值
    dtMax  = 0.2;                       % 绝对最大步长（防漏检的保险）

    % --------- 初始化 ---------
    t = t0;
    dt = min(dtInit, dtMax);
    intervals = [];

    % 计算遮挡状态函数（直接调用 isCylinderBlockedBySphere）
    function occ = isOccAt(time)
        obsPos = init.observerPos + time * init.observerVel;
        cylC   = init.cylinderCenter + time * init.cylinderVel;
        sphC   = init.sphereCenter   + time * init.sphereVel;
        occ = isCylinderBlockedBySphere( ...
            obsPos, ...
            cylC, init.cylinderRadius, init.cylinderHeight, init.cylinderDir, ...
            sphC, init.sphereRadius);
    end

    % 在给定区间内（二端异态）二分定位切换时刻
    function tc = refineSwitchWithin(ta, tb, occA)
        while (tb - ta) > tolRef
            tm = 0.5 * (ta + tb);
            if isOccAt(tm) == occA
                ta = tm;
            else
                tb = tm;
            end
        end
        tc = tb;
    end

    occPrev = isOccAt(t);
    tMark = t;  % 当前遮挡段的起点（当处于遮挡时）

    % --------- 主循环：先快后慢，自适应步长 ---------
    while t < t1 - eps
        % 限制不越界且不超过最大步长
        dt = min([dt, t1 - t, dtMax]);
        occNext = isOccAt(t + dt);

        if occNext == occPrev
            % 中点守卫：避免“首尾同态但中途发生遮挡”的整步漏检
            tm = t + 0.5 * dt;
            occMid = isOccAt(tm);
            if occMid ~= occPrev
                % 在前半步内二分定位第一次切换时刻
                tc = refineSwitchWithin(t, tm, occPrev);
                if occPrev
                    % 遮挡 -> 非遮挡：收尾一段遮挡区间
                    intervals = [intervals; tMark, tc]; %#ok<AGROW>
                else
                    % 非遮挡 -> 遮挡：开始记录
                    tMark = tc;
                end
                % 在事件点落地并缩小步长继续
                t = tc;
                occPrev = ~occPrev;
                dt = max(dt * shrink, dtMin);
                continue; % 重新进入循环
            end

            % 未检测到中点变化：接受步长、尝试加速
            t = t + dt;
            if occPrev
                % 遮挡持续，等待结束再记录区间
            end
            dt = min(dt * grow, min(t1 - t0, dtMax));
        else
            % 发生遮挡状态切换：在 [t, t+dt] 内二分定位切换时刻
            ta = t; tb = t + dt;
            while (tb - ta) > tolRef
                tm = 0.5 * (ta + tb);
                if isOccAt(tm) == occPrev
                    ta = tm;
                else
                    tb = tm;
                end
            end
            tc = tb; % 切换时刻

            if occPrev
                % 遮挡 -> 非遮挡：收尾一段遮挡区间
                intervals = [intervals; tMark, tc]; %#ok<AGROW>
            else
                % 非遮挡 -> 遮挡：开始记录
                tMark = tc;
            end

            % 在事件点落地并缩小步长继续
            t = tc;
            occPrev = ~occPrev;
            dt = max(dt * shrink, dtMin);
        end
    end

    % 收尾：若结束时仍遮挡，补齐到 t1
    if occPrev
        intervals = [intervals; tMark, t1];
    end

    % 合并相邻小段
    intervals = mergeSmallGaps(intervals, 1e-12);

    if isempty(intervals)
        totalDuration = 0;
    else
        totalDuration = sum(intervals(:,2) - intervals(:,1));
    end
end

function intervals = mergeSmallGaps(intervals, tol)
    if isempty(intervals), return; end
    intervals = sortrows(intervals, 1);
    merged = intervals(1, :);
    for i = 2:size(intervals,1)
        last = merged(end, :);
        cur = intervals(i, :);
        if cur(1) <= last(2) + tol
            merged(end, 2) = max(last(2), cur(2));
        else
            merged = [merged; cur]; %#ok<AGROW>
        end
    end
    intervals = merged;
end 
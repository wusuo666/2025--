function [totalDuration, intervals] = computeOcclusionFixedStep(t0, t1, dt, init)
% 固定时间步长扫描遮挡：仅按固定 dt 逐步判断遮挡，不做任何细化或额外处理
% 直接调用 isCylinderBlockedBySphere 进行遮挡判定（以每步左端点为准）
%
% 输入：
%   t0, t1 : 起止时间（秒）
%   dt     : 固定步长（秒），必须 > 0
%   init   : 初始条件结构体：
%            .observerPos, .observerVel
%            .cylinderCenter, .cylinderVel, .cylinderRadius, .cylinderHeight, .cylinderDir
%            .sphereCenter, .sphereVel, .sphereRadius
%
% 输出：
%   totalDuration : 遮挡总时长（秒），按离散步长求和
%   intervals     : n×2 遮挡区间 [tbeg, tend]，按左端点遮挡的连续步拼接

    % -------- 参数与边界检查 --------
    if dt <= 0
        error('dt 必须为正数');
    end
    if t1 <= t0
        totalDuration = 0;
        intervals = [];
        return;
    end

    % -------- 遮挡状态查询（基于左端点） --------
    function occ = isOccAt(time)
        obsPos = init.observerPos + time * init.observerVel;
        cylC   = init.cylinderCenter + time * init.cylinderVel;
        sphC   = init.sphereCenter   + time * init.sphereVel;
        occ = isCylinderBlockedBySphere( ...
            obsPos, ...
            cylC, init.cylinderRadius, init.cylinderHeight, init.cylinderDir, ...
            sphC, init.sphereRadius);
    end

    % -------- 固定步长扫描 --------
    nSteps = ceil((t1 - t0) / dt);
    intervals = [];
    inOcc = false;
    totalDuration = 0;

    for k = 0:(nSteps-1)
        ti = t0 + k * dt;
        if ti >= t1
            break;
        end
        tj = min(ti + dt, t1); % 该步右端，可能为最后一段的短步

        occ = isOccAt(ti);     % 以左端点为准

        if occ
            if ~inOcc
                segStart = ti; %#ok<NASGU>
                inOcc = true;
            end
            % 该步贡献的遮挡时长
            totalDuration = totalDuration + (tj - ti);
        else
            if inOcc
                % 遮挡刚在本步开始前结束，上一段结束于 ti
                intervals = [intervals; segStart, ti]; %#ok<AGROW>
                inOcc = false;
            end
        end
    end

    % 收尾：若结束时仍处于遮挡，将最后一段补到 t1
    if inOcc
        intervals = [intervals; segStart, t1];
    end
end 
import React from 'react';

const StatCard = ({ label, value, subtext, icon: Icon, trend }) => {
  return (
    <div className="stat-card">
      <div className="stat-card-top">
        <span className="stat-card-label">{label}</span>
        {Icon && (
          <div className="stat-card-icon-wrapper">
            <Icon size={18} />
          </div>
        )}
      </div>
      <div className="stat-card-value">{value}</div>
      {(subtext || trend) && (
        <div className="stat-card-footer">
          {trend && <span className="stat-trend">{trend}</span>}
          {subtext && <span>{subtext}</span>}
        </div>
      )}
    </div>
  );
};

export default StatCard;

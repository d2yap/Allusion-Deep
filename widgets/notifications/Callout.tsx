import React from 'react';

interface ICallout {
  icon: React.ReactNode;
  header?: string | number;
  children: React.ReactNode;
}

export const Callout = ({ icon, header, children }: ICallout) => {
  return (
    <div className="callout">
      <span className="callout-icon">{icon}</span>
      <span className="callout-header">{header}</span>
      <div className="callout-body">{children}</div>
    </div>
  );
};

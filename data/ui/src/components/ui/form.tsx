import { InputNumber, InputNumberProps } from 'antd';
import React from 'react';

interface ContentRowProps {
  label: string;
  content: React.ReactNode;
  suffix?: React.ReactNode;
}

export function ContentRow(props: ContentRowProps): React.ReactElement {
  const { label, content, suffix } = props;

  return (
    <div className='min-w-full flex items-center justify-between mb-2 last:mb-0 font-light'>
      <span>{label}</span>
      <div className='flex items-center'>
        {content}
        {suffix && <div className='flex ml-2'>{suffix}</div>}
      </div>
    </div>
  );
}

interface IndicatorLabelProps {
  name: string;
  description: string;
}

export function IndicatorLabel(props: IndicatorLabelProps): React.ReactElement {
  const { name, description } = props;

  return (
    <span className='ml-2 font-light'>
      {name}
      <span className='ml-1 italic font-light text-dark-gray'>
        ({description})
      </span>
    </span>
  );
}

export function NumberInput(
  props: InputNumberProps<number>
): React.ReactElement {
  return <InputNumber className='w-16 text-dark-gray' {...props} />;
}

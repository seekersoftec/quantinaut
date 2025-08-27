import { Table } from 'antd';
import dateFormat from 'dateformat';
import React from 'react';
import { GetTradesResponse, Trade } from '../../api/trades';
import { ApiQueryResult } from '../../api/types';
import { paginationProps } from '../../utils/pagination';
import { ExportButton } from '../buttons/export';
import * as animated from '../ui/animated';
import { Header } from '../ui/header';
import { Loader } from '../ui/loader';

interface TradesTableProps extends ApiQueryResult<GetTradesResponse> {}

export function TradesTable(props: TradesTableProps): React.ReactElement {
  const { data, loading } = props;

  if (!data || loading) {
    return <Loader />;
  }

  const columns = [
    {
      title: 'Id',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
    },
    {
      title: 'Entry',
      dataIndex: 'entry',
      key: 'entry',
    },
    {
      title: 'Exit',
      dataIndex: 'exit',
      key: 'exit',
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      render: (quantity: Trade['quantity']) => quantity.toFixed(4),
    },
    {
      title: 'Time',
      dataIndex: 'time',
      key: 'time',
      render: (time: Trade['time']) => dateFormat(time, 'mmm dS hh:MM tt'),
    },
  ];

  const extra: React.ReactNode[] = React.Children.toArray([
    <ExportButton type='trades' data={data.trades} />,
  ]);

  return (
    <animated.Div className='mt-4 w-full flex flex-1 flex-col table'>
      <Header title='Trades' subtitle='Trades executed so far' extra={extra} />
      <Table
        className='text-xs font-light'
        pagination={paginationProps(data.trades.length)}
        columns={columns}
        dataSource={data.trades}
      />
    </animated.Div>
  );
}

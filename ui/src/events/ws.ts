import WebSocket from 'ws';
import { API_HOST } from '../config/api';
import Env from '../utils/env';
import * as Notifications from '../utils/notifications';
import { Events } from './types';

const SERVER_URL = `ws://${API_HOST}/nats`;

export class PubSub {
  private static instance: PubSub;
  private ws: WebSocket;
  private subscribers: Map<Events, ((data: any) => void)[]>;

  private constructor(ws: WebSocket) {
    this.ws = ws;
    this.subscribers = new Map();
    this.ws.onmessage = this.handleMessage;
    this.ws.onclose = this.handleClose;
    this.ws.onerror = this.handleError;
  }

  static getInstance = async (): Promise<PubSub> => {
    if (!this.instance) {
      try {
        const ws = new WebSocket(SERVER_URL, {
          headers: {
            'Authorization': `Basic ${btoa(`${Env.WS_USER}:${Env.WS_PASS}`)}`
          }
        });

        await new Promise<void>((resolve, reject) => {
          ws.onopen = () => {
            console.log('WebSocket connection established');
            resolve();
          };
          ws.onerror = (event) => {
            reject(new Error('WebSocket connection error'));
          };
        });

        this.instance = new PubSub(ws);
      } catch (err) {
        Notifications.error('PubSub get instance', err);
        throw err;
      }
    }
    return this.instance;
  };

  private handleMessage = (event: WebSocket.MessageEvent) => {
    try {
      const { event: eventName, data } = JSON.parse(event.data.toString());
      const callbacks = this.subscribers.get(eventName);
      if (callbacks) {
        callbacks.forEach(cb => cb(data));
      }
    } catch (err) {
      console.error('Failed to parse message or execute callback', err);
    }
  };

  private handleClose = () => {
    console.log('WebSocket connection closed');
    // You might want to implement a reconnection strategy here
  };

  private handleError = (event: WebSocket.ErrorEvent) => {
    Notifications.error('WebSocket error', event.error);
  };

  subscribe = <T>(event: Events, cb: (data: T) => void): void => {
    if (!this.subscribers.has(event)) {
      this.subscribers.set(event, []);
    }
    this.subscribers.get(event)?.push(cb as (data: any) => void);
  };

  // The request method is fundamentally different with a generic WebSocket
  // as there is no built-in request/reply pattern. This implementation
  // is a simple placeholder and would need a server-side equivalent
  // to work properly (e.g., using a message ID for correlation).
  request = async <T = unknown, D = unknown>(
    event: Events,
    data?: D
  ): Promise<T> => {
    return new Promise<T>((resolve, reject) => {
      // Send a request message
      this.ws.send(JSON.stringify({ type: 'request', event, data }));

      // This is a simplified example. A real implementation would need
      // to correlate the response with the request, e.g., using a unique ID.
      const onResponse = (response: any) => {
        if (response.event === event) {
          resolve(response.data as T);
          // Cleanup the listener
          this.ws.off('message', onResponse);
        }
      };
      this.ws.on('message', onResponse);
    });
  };
}
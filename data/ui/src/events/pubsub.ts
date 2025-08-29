import { io, Socket } from "socket.io-client";
import Env from '../utils/env';
import * as Notifications from '../utils/notifications';
import { Events } from './types';

const SERVER_URL = `${Env.API_HOST}/ws`;

const AUTH = {
  user: Env.API_USER,
  pass: Env.API_PASS,
};

// Define event types
interface ResponseHandler {
  response?: (data: unknown) => void;
}

type ServerToClientEvents = {
  [event: string]: (data: any) => void;
} & ResponseHandler;


interface ClientToServerEvents {
  [event: string]: (data?: any, callback?: (response: unknown) => void) => void;
}

export class PubSub {
  private static instance: PubSub;
  private conn: Socket<ServerToClientEvents, ClientToServerEvents>;

  // Private constructor
  private constructor(conn: Socket) {
    this.conn = conn;
  }

  static getInstance = async (): Promise<PubSub> => {
      if (!this.instance) {
        try {
              const conn = io(SERVER_URL, {
                auth: { 
                  token: `${AUTH.user}:${AUTH.pass}`, // Basic Auth token 
                 }
              });

              conn.on("connect", () => {
                console.log("Connected to Socket.IO server");
                Notifications.success('Server connected');
              });

              conn.on("disconnect", (reason) => {
                console.log("Disconnected:", reason);
                Notifications.error('Server disconnected', new Error(reason));
              });

              conn.on("connect_error", (err) => {
                console.error("Connection error:", err);
                if (err.message === "unauthorized" || err.message === "Authentication failed") {
                  Notifications.error("Authentication failed...", err);
                  // Maybe redirect...
                } else {
                  Notifications.error('Server connection error', err);
                }
              });
  
          this.instance = new PubSub(conn);
        } catch (err) {
          Notifications.error('PubSub get instance', err);
        }
      }
  
      return this.instance;
  };


  // Subscribe to server events
  public subscribe<T>(event: Events, cb: (data: T) => void): void {
    this.conn.on(event, (data: T) => {
      cb(data);
    });
  }

  // Unsubscribe from server events
  public unsubscribe(event: Events): void {
    this.conn.off(event);
  }

  // Request-response: client sends an event and expects a reply
  public request<T = unknown>(event: Events, data?: any): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.conn.emit(event, data, (response: unknown) => {
        resolve(response as T);
      });
      // Optionally add timeout handling here...
    });
  }

  // Publish-only: fire-and-forget
  public publish(event: Events, data?: any): void {
    this.conn.emit(event, data);
  }
}



// const pubsub = PubSub.getInstance("http://localhost:3000");

// // Subscribe to event
// pubsub.subscribe("news", (data) => {
//   console.log("Received news:", data);
// });

// // Publish an event
// pubsub.publish("update", { version: "1.0.1" });

// // Request-response
// pubsub.request("getStatus", {}).then((status) => {
//   console.log("Server status:", status);
// });

// room-based messaging, namespaces,
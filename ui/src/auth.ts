// import { gql } from "graphql-request";

import NextAuth from "next-auth";
import Credentials from "next-auth/providers/credentials";
import Discord from "next-auth/providers/discord";
import Google from "next-auth/providers/google"; 
import GitHub from "next-auth/providers/github"

type AuthCredentials = {
  username: string;
  password: string;
};

const serverAPIRequest = async (variables: AuthCredentials) => {
  // const query = gql`
  //   query users($email: String!, $password: bpchar!) {
  //     users(where: { email: { _eq: $email }, password: { _eq: $password } }) {
  //       id
  //       name
  //       email
  //       image
  //       created_at
  //       updated_at
  //     }
  //   }
  // `;

  console.log(variables);

  const BACKEND_URL = process.env.AUTH_BACKEND_URL || "http://127.0.0.1:8000";
  const res = await fetch(BACKEND_URL + "/api/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // "x-hasura-admin-secret": process.env.AUTH_HASURA_SECRET!,
    },
    // body: JSON.stringify({ query, variables }),
    body: JSON.stringify(variables)
  });

  console.log(await res.json());
  if (!res.ok) {
    throw new Error("Failed to fetch token");
  }

  return await res.json();
};

export const { auth, handlers, signIn, signOut } = NextAuth({
  providers: [
    Credentials({
      id: "nautilus-server-credentials",
      name: "Nautilus Server Credentials",
      credentials: {
        username: { label: "Username", type: "name" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        try {
          if (!credentials) {
            return null;
          }
   
          const { data } = await serverAPIRequest({
            username: credentials.username as string,
            password: credentials.password as string,
          });

          if (data.length > 0) {
            return {
              id: data.uid,
              name: data.uid,
              // email: data.users[0].email,
              // image: data.users[0].image??,
              token: data.access_token
            };
          } else {
            return null;
          }
        } catch (error) {
          throw new Error(
            JSON.stringify({ errors: "Authorize error", status: false })
          );
        }
      },
    }),
    Discord({
      clientId: process.env.AUTH_DISCORD_ID,
      clientSecret: process.env.AUTH_DISCORD_SECRET,
    }),
    Google({
      clientId: process.env.AUTH_AUTH_GOOGLE_ID,
      clientSecret: process.env.AUTH_AUTH_GOOGLE_SECRET,
    }),
    GitHub, 
  ],
  pages: {
    signIn: "/auth/signin",
    signOut: "/auth/signout",
  },
  session: { strategy: "jwt" },
  callbacks: {
    async signIn(userDetail) {
      if (Object.keys(userDetail).length === 0) {
        return false;
      }
      return true;
    },
    async redirect({ baseUrl }) {
      return `${baseUrl}/`;
    },
    async session({ session, token }) {
      if (session.user?.name) session.user.name = token.name;
      return session;
    },
    async jwt({ token, user }) {
      let newUser = { ...user } as any;
      if (newUser.first_name && newUser.last_name)
        token.name = `${newUser.first_name} ${newUser.last_name}`;
      return token;
    },
  },
});

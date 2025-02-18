'use client';

import ChatInterface from '../components/ChatInterface';

export default function Home() {
  return (
    <div className="min-h-screen bg-[#343541]">
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-3xl mx-auto">
          <ChatInterface />
        </div>
      </main>
    </div>
  );
}

'use client';

import { useState } from 'react';
import FileUpload from '../components/FileUpload';
import ChatInterface from '../components/ChatInterface';

export default function Home() {
  const [isDataLoaded, setIsDataLoaded] = useState(false);

  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">
          Financial Data Assistant
        </h1>
        
        <div className="grid md:grid-cols-[350px_1fr] gap-8">
          {/* Left Panel - File Upload */}
          <div className="space-y-4">
            <FileUpload onDataLoaded={() => setIsDataLoaded(true)} />
            <div className="p-4 bg-white rounded-lg shadow">
              <h2 className="font-semibold mb-2">Supported Files:</h2>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• CSV - Financial Data</li>
                <li>• PDF - Financial Reports</li>
                <li>• PPT - Financial Presentations</li>
              </ul>
            </div>
          </div>

          {/* Right Panel - Chat Interface */}
          <div>
            <ChatInterface isDataLoaded={isDataLoaded} />
          </div>
        </div>
      </div>
    </main>
  );
}

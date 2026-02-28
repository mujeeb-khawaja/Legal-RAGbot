import React, { useState, useRef, useEffect } from 'react';
import { Send, Scale, Plus, History, Loader2, LayoutPanelLeft, ChevronRight, User, Bot, AlertCircle, Settings2, Edit, Trash2, Check, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { sendMessage } from './services/api';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const STORAGE_KEY = 'afghan_legal_chats';
const EXPIRY_TIME = 24 * 60 * 60 * 1000; // 24 hours

function App() {
  const [chats, setChats] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return [];
    try {
      const parsed = JSON.parse(saved);
      // Filter out expired chats
      return parsed.filter(chat => Date.now() - chat.timestamp < EXPIRY_TIME);
    } catch {
      return [];
    }
  });

  const [activeChatId, setActiveChatId] = useState(() => {
    return chats.length > 0 ? chats[0].id : null;
  });

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  const messagesEndRef = useRef(null);

  const activeChat = chats.find(c => c.id === activeChatId);
  const messages = activeChat ? activeChat.messages : [];

  // Sync to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  }, [chats]);

  // Initial chat creation if empty
  useEffect(() => {
    if (chats.length === 0) {
      createNewChat();
    } else if (!activeChatId) {
      setActiveChatId(chats[0].id);
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const createNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      title: 'New Consultation',
      timestamp: Date.now(),
      messages: [
        {
          role: 'assistant',
          content: 'Hello! I am your Afghan Legal Expert. How can I assist you with legal queries today?',
          type: 'text'
        }
      ]
    };
    setChats([newChat, ...chats]);
    setActiveChatId(newChat.id);
  };

  const deleteChat = (e, id) => {
    e.stopPropagation();
    const filtered = chats.filter(c => c.id !== id);
    setChats(filtered);
    if (activeChatId === id) {
      setActiveChatId(filtered.length > 0 ? filtered[0].id : null);
    }
  };

  const startRenaming = (e, chat) => {
    e.stopPropagation();
    setEditingChatId(chat.id);
    setEditTitle(chat.title);
  };

  const saveRename = (e) => {
    e.stopPropagation();
    setChats(chats.map(c =>
      c.id === editingChatId ? { ...c, title: editTitle } : c
    ));
    setEditingChatId(null);
  };

  const cancelRename = (e) => {
    e.stopPropagation();
    setEditingChatId(null);
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading || !activeChatId) return;

    const userMessage = { role: 'user', content: input, type: 'text' };

    // Auto-rename chat and remove initial greeting if it's the first user message
    const isFirstUserMessage = messages.length === 1 && messages[0].role === 'assistant';
    const updatedTitle = isFirstUserMessage
      ? (input.length > 30 ? input.substring(0, 30) + '...' : input)
      : activeChat.title;

    setChats(chats.map(c =>
      c.id === activeChatId
        ? { ...c, title: updatedTitle, messages: isFirstUserMessage ? [userMessage] : [...c.messages, userMessage] }
        : c
    ));

    setInput('');
    setIsLoading(true);

    try {
      const data = await sendMessage(input);

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        type: 'ai'
      };

      setChats(prev => prev.map(c =>
        c.id === activeChatId
          ? { ...c, messages: [...c.messages, assistantMessage] }
          : c
      ));
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: '**Error:** Failed to connect to the legal database.',
        type: 'error'
      };
      setChats(prev => prev.map(c =>
        c.id === activeChatId
          ? { ...c, messages: [...c.messages, errorMessage] }
          : c
      ));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#212121] text-[#ececec] overflow-hidden font-sans">
      {/* Sidebar - GPT Grey */}
      <div className="hidden md:flex flex-col w-[260px] bg-[#171717] border-r border-[#303030]">
        <div className="p-4 flex items-center justify-between mb-2">
          <button
            onClick={createNewChat}
            className="flex-1 flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-[#2f2f2f] transition-all text-sm font-medium border border-[#303030]"
          >
            <Plus className="w-4 h-4" /> New consultation
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 space-y-1">
          <div className="text-[11px] font-bold text-[#676767] uppercase tracking-wider mb-2 px-3">History</div>
          {chats.map(chat => (
            <div
              key={chat.id}
              onClick={() => setActiveChatId(chat.id)}
              className={cn(
                "group flex items-center gap-2 p-2 rounded-lg cursor-pointer text-sm overflow-hidden transition-all",
                activeChatId === chat.id ? "bg-[#2f2f2f] text-white" : "text-[#ececec] hover:bg-[#2f2f2f]/50"
              )}
            >
              <History className="w-4 h-4 flex-shrink-0 text-[#676767]" />
              <div className="flex-1 truncate">
                {editingChatId === chat.id ? (
                  <input
                    autoFocus
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') saveRename(e);
                      if (e.key === 'Escape') cancelRename(e);
                    }}
                    onBlur={saveRename}
                    className="bg-transparent border-none outline-none w-full text-white text-[13px]"
                  />
                ) : (
                  <span className="text-[13px]">{chat.title}</span>
                )}
              </div>

              <div className={cn(
                "flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity",
                activeChatId === chat.id && "opacity-100"
              )}>
                {editingChatId === chat.id ? (
                  <Check onClick={saveRename} className="w-3.5 h-3.5 text-green-500 hover:text-green-400" />
                ) : (
                  <>
                    <Edit onClick={(e) => startRenaming(e, chat)} className="w-3.5 h-3.5 text-[#676767] hover:text-white" />
                    <Trash2 onClick={(e) => deleteChat(e, chat.id)} className="w-3.5 h-3.5 text-[#676767] hover:text-red-400" />
                  </>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-[#303030]">
          <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-[#2f2f2f] cursor-pointer transition-all">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-xs font-bold text-white uppercase italic">AF</div>
            <div className="flex-1 overflow-hidden">
              <div className="text-[13px] font-medium truncate">Legal Expert AI</div>
              <div className="text-[10px] text-[#b4b4b4] tracking-widest uppercase font-bold leading-none mt-1">Law Dept</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative w-full h-full">
        {/* Header - Minimal GPT style */}
        <div className="h-14 flex items-center justify-between px-6 border-b border-[#303030] bg-[#212121]/80 backdrop-blur-sm z-10">
          <div className="flex items-center gap-2">
            <Scale className="w-5 h-5 text-[#b4b4b4]" />
            <span className="font-semibold text-sm">AfghanLegal v1.0</span>
          </div>
          <div className="flex items-center gap-4">
            <Settings2 className="w-4 h-4 text-[#676767] cursor-pointer hover:text-white transition-all" />
          </div>
        </div>

        {/* Chat Area - Wide Layout */}
        <div className="flex-1 overflow-y-auto pb-36">
          {messages.length === 1 && (
            <div className="h-full flex flex-col items-center justify-center text-center p-8 max-w-2xl mx-auto">
              <div className="w-16 h-16 bg-[#2f2f2f] border border-[#303030] rounded-2xl flex items-center justify-center mb-6">
                <Scale className="w-8 h-8 text-[#ececec]" />
              </div>
              <h2 className="text-2xl font-semibold text-white mb-6">How can I help you today?</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
                <button onClick={() => { setInput('What are the rules regarding judicial independence?'); handleSend(); }} className="p-4 text-left text-[13px] bg-transparent border border-[#303030] rounded-xl hover:bg-[#2f2f2f] transition-all text-[#b4b4b4] hover:text-[#ececec]">Rule of Law & Independence</button>
                <button onClick={() => { setInput('Explain the citizenship laws in the current context.'); handleSend(); }} className="p-4 text-left text-[13px] bg-transparent border border-[#303030] rounded-xl hover:bg-[#2f2f2f] transition-all text-[#b4b4b4] hover:text-[#ececec]">Citizenship & Residence Laws</button>
              </div>
            </div>
          )}

          <div className="w-full max-w-[800px] mx-auto pt-8">
            {messages.map((m, i) => (
              <div key={i} className={cn(
                "group w-full py-8 border-b border-[#303030]/30",
                m.role === 'user' ? "bg-transparent" : "bg-transparent"
              )}>
                <div className="flex gap-6 max-w-[800px] mx-auto px-4 sm:px-4">
                  <div className={cn(
                    "w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center mt-1 border",
                    m.role === 'user' ? "bg-[#2f2f2f] border-[#303030]" : "bg-blue-600 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.3)]"
                  )}>
                    {m.role === 'user' ? <User className="w-4 h-4 text-slate-300" /> : <Bot className="w-4 h-4 text-white" />}
                  </div>

                  <div className="flex-1 flex flex-col gap-2 overflow-hidden">
                    <div className="prose prose-invert prose-base max-w-none leading-relaxed text-[#d1d1d1]">
                      <ReactMarkdown>
                        {m.content}
                      </ReactMarkdown>
                    </div>

                    {/* Source Citations - Accordion Style */}
                    {m.sources && m.sources.length > 0 && (
                      <div className="mt-4 space-y-3">
                        <div className="flex items-center gap-2 text-[10px] font-bold text-[#676767] uppercase tracking-[0.2em] mb-4">
                          <LayoutPanelLeft className="w-3 h-3" /> Supporting Legal Context
                        </div>
                        <div className="space-y-2">
                          {m.sources.map((src, j) => (
                            <details key={j} className="group border border-[#303030] rounded-xl bg-[#171717]/50 overflow-hidden hover:bg-[#171717] transition-all">
                              <summary className="p-3 text-[12px] font-medium cursor-pointer flex items-center justify-between text-[#b4b4b4] select-none hover:text-[#ececec]">
                                <span className="flex items-center gap-3">
                                  <div className="w-1 h-1 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
                                  Document Evidence Room #{j + 1}
                                </span>
                                <ChevronRight className="w-3 h-3 group-open:rotate-90 transition-transform" />
                              </summary>
                              <div className="p-5 text-[12px] text-[#999] bg-[#0a0a0a] border-t border-[#303030] leading-relaxed font-serif italic">
                                "{src}"
                              </div>
                            </details>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="w-full py-8 border-b border-[#303030]/30 transition-all opacity-70">
                <div className="flex gap-6 max-w-[800px] mx-auto px-4">
                  <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center border border-blue-500 shadow-lg">
                    <Loader2 className="w-4 h-4 animate-spin text-white" />
                  </div>
                  <div className="text-[14px] text-[#676767] italic animate-pulse py-2">
                    Thinking...
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} className="h-4" />
          </div>
        </div>

        {/* Input Area - Centered & Wide */}
        <div className="absolute bottom-0 left-0 right-0 p-4 sm:p-6 bg-gradient-to-t from-[#212121] via-[#212121] to-transparent z-20">
          <div className="max-w-[800px] mx-auto relative group">
            <div className="relative flex items-center bg-[#2f2f2f] border border-[#303030] rounded-2xl p-1.5 focus-within:border-[#4d4d4d] transition-all shadow-xl">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask about legal procedures or document verification..."
                className="flex-1 bg-transparent text-[#ececec] pl-4 pr-12 py-3 focus:outline-none placeholder:text-[#676767] text-[15px]"
              />
              <button
                onClick={handleSend}
                disabled={isLoading || !input.trim()}
                className="p-2.5 bg-white text-black rounded-xl hover:bg-[#d7d7d7] disabled:bg-[#303030] disabled:text-[#676767] transition-all"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
            <p className="text-center text-[11px] text-[#676767] mt-3 tracking-wide">
              Chats will be removed after 24 hours only. Chat can make mistakes, please verify the answers.
            </p>
          </div>
        </div>
      </div>
    </div >
  );
}

export default App;
